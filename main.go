package main

import (
	"flag"
	"fmt"
	"image"
	"log"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"

	// Specific image-types are not used explicitly in the code below,
	// but importing registers them for use with image.Decode.
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"

	"gitlab.com/oelmekki/matrix"
)

// Config contains all command-line options.
type Config struct {
	SourceFile, OutputFile string
	Space                  string
	Flat, BT470            bool
}

type PixelQuantizer func(FloatNRGBA) color.NRGBA
type QuantizeBuilder func([]int, *dither, Config) PixelQuantizer
type LevelsValue [4]float64
type ErrorValue [4]float64
type Errors [2][]ErrorValue

// FloatNRGBA is an RGB color with straight alpha, all normalized (0.0 to 1.0),
// and all using float64 because that's what math does.
type FloatNRGBA struct {
	R, G, B, A float64
}

type dither struct {
	x int
	e Errors
}

var fConfig = Config{}

func init() {
	flag.StringVar(&fConfig.SourceFile, "in", "img.png", "Source image to process")
	flag.StringVar(&fConfig.OutputFile, "out", "output.png", "Output image file name")
	flag.StringVar(&fConfig.Space, "space", "Y4", "Color space and quantization to use, e.g. RGB332")
	flag.BoolVar(&fConfig.Flat, "flat", false, "Skip dithering")
	flag.BoolVar(&fConfig.BT470, "sd", false, "Use SDTV BT.470 luma weights instead of BT.703")
}

// FMUL is the factor between color.Color and normalized floats
const FMUL = 65535.0

// FMUL8 is the factor between normalized floats and 8-bit representations,
// such as color.NRGBA
const FMUL8 = 255.0

// straightAlphaFloat turns a color.Color into normalized float64 components
// with straight (not premultiplied) alpha.
func straightAlphaFloat(c color.Color) FloatNRGBA {
	r, g, b, a := c.RGBA()
	aDiv := float64(a)

	// for the RGB components, FMUL cancels out: (r/FMUL) / (a/FMUL) = r/a
	return FloatNRGBA{
		R: float64(r) / aDiv,
		G: float64(g) / aDiv,
		B: float64(b) / aDiv,
		A: aDiv / FMUL,
	}
}

// diffuseFloydSteinberg implements the error diffusion algorithm on a window
// of rows.
func diffuseFloydSteinberg(errorRows *Errors, ox int, errVec ErrorValue) {
	xMax := len(errorRows[0]) - 1
	for i, v := range errVec {
		// "v != v" is inlined math.isNaN() to avoid breaking inline-ability
		if v == 0.0 || v != v {
			continue
		}

		if ox > 0 {
			errorRows[1][ox-1][i] += v * (3.0 / 16)
		}
		errorRows[1][ox][i] += v * (5.0 / 16)
		if ox < xMax {
			errorRows[0][ox+1][i] += v * (7.0 / 16)
			errorRows[1][ox+1][i] += v * (1.0 / 16)
		}
	}
}

func levelsFromBitsSingle(bits int) float64 {
	if bits < 0 {
		panic("negative bits requested")
	}

	levels := (1 << bits) - 1
	return float64(levels)
}

func levelsFromBits(bits []int, space string) (levels LevelsValue) {
	if len(bits) < 3 {
		what := fmt.Sprintf("Not enough bits for %s channels (needed 4, got %d", space, len(bits))
		panic(what)
	}

	end := len(bits)
	if end > len(levels) {
		end = len(levels)
	}
	for i := 0; i < end; i += 1 {
		levels[i] = levelsFromBitsSingle(bits[i])
	}

	return
}

// normQuantize quantizes a float to levels in the 0.0-1.0 range.  vErr is the
// value with error added; levels is the number of quantization levels, or zero
// for no quantization.
func normQuantize(vErr float64, levels float64) (out float64) {
	if levels == 0.0 {
		// no levels => passthru
		out = vErr
	} else if levels == 1 {
		// 1 level = "black or white"
		out = math.RoundToEven(vErr)
	} else {
		// multiple levels; do full calculation
		out = math.RoundToEven(levels*vErr) / levels
	}

	out = math.Min(1.0, math.Max(0.0, out))
	return
}

// limitQuantize quantizes a float to levels in an arbitrary range.  vErr is the
// value with error added; levels is the number of quantization levels, or zero
// for no quantization.
func limitQuantize(vErr, min, max float64, levels float64) (out float64) {
	if levels == 0.0 {
		out = vErr
	} else if levels == 1 {
		out = math.RoundToEven(vErr)
	} else {
		out = math.RoundToEven(levels*vErr) / levels
	}

	out = math.Min(max, math.Max(min, out))
	return
}

func int8OfFloat(value float64) uint8 {
	if value <= 0.0 {
		return 0
	}
	if value > (1.0 - 1/256) {
		return 255
	}

	return uint8(value * FMUL8)
}

func quantizerGray(bits []int, ctx *dither, config Config) PixelQuantizer {
	if len(bits) < 1 {
		panic("not enough bits specified for Y channel")
	}

	levels := [2]float64{
		levelsFromBitsSingle(bits[0]),
		0.0,
	}
	if len(bits) > 1 {
		levels[1] = levelsFromBitsSingle(bits[1])
	}

	return func(clr FloatNRGBA) color.NRGBA {
		// compute the full-precision grayscale
		var y0 float64
		if config.BT470 {
			y0 = float64(0.299*clr.R + 0.587*clr.G + 0.114*clr.B) // BT.470
		} else {
			y0 = float64(0.2126*clr.R + 0.7152*clr.G + 0.0722*clr.B) // BT.709
		}

		// quantize
		ce := ctx.e[0][ctx.x] // current error
		yF := normQuantize(y0+ce[0], levels[0])
		aF := normQuantize(clr.A+ce[1], levels[1])
		if !config.Flat {
			diffuseFloydSteinberg(&ctx.e, ctx.x, ErrorValue{y0 - yF, clr.A - aF})
		}

		// convert to output space
		y8 := int8OfFloat(yF)
		return color.NRGBA{R: y8, G: y8, B: y8, A: uint8(aF * FMUL8)}
	}
}

func quantizerRgb(bits []int, ctx *dither, config Config) PixelQuantizer {
	levels := levelsFromBits(bits, "RGB[A]")

	return func(clr FloatNRGBA) color.NRGBA {
		ce := ctx.e[0][ctx.x]

		rF := normQuantize(clr.R+ce[0], levels[0])
		gF := normQuantize(clr.G+ce[1], levels[1])
		bF := normQuantize(clr.B+ce[2], levels[2])
		aF := normQuantize(clr.A+ce[3], levels[3])

		if !config.Flat {
			eVal := ErrorValue{clr.R - rF, clr.G - gF, clr.B - bF, clr.A - aF}
			diffuseFloydSteinberg(&ctx.e, ctx.x, eVal)
		}

		return color.NRGBA{
			R: int8OfFloat(rF),
			G: int8OfFloat(gF),
			B: int8OfFloat(bF),
			A: uint8(aF * FMUL8),
		}
	}
}

func modPositively(n, d float64) float64 {
	v := math.Mod(n, d)
	if v < 0.0 {
		v += d
	} else if v >= d {
		v -= d
	}

	return v
}

func getHue6(clr FloatNRGBA, cMax, delta float64) (h float64) {
	// hue, based on the max channel; or 0.0 if delta is 0.0
	// h runs 0.0 to 6.0 (a factor of 60 is ignored in both directions)
	if delta != 0.0 {
		if cMax == clr.R {
			h = modPositively((clr.G-clr.B)/delta, 6.0)
		} else if cMax == clr.G {
			h = ((clr.B - clr.R) / delta) + 2
		} else {
			// implied: cMax == clr.B
			h = ((clr.R - clr.G) / delta) + 4
		}
	}

	return
}

func rgbFromHCXM(h, C, X, M float64) (r, g, b float64) {
	// intermediate values
	var rP, gP, bP float64

	if h < 1.0 {
		rP, gP = C, X
	} else if h < 2.0 {
		rP, gP = X, C
	} else if h < 3.0 {
		gP, bP = C, X
	} else if h < 4.0 {
		gP, bP = X, C
	} else if h < 5.0 {
		bP, rP = C, X
	} else {
		bP, rP = X, C
	}

	r = rP + M
	g = gP + M
	b = bP + M

	return
}

func quantizerHsv(bits []int, ctx *dither, config Config) PixelQuantizer {
	levels := levelsFromBits(bits, "HSV[A]")

	return func(clr FloatNRGBA) color.NRGBA {
		ce := ctx.e[0][ctx.x]

		// convert to HSV
		cMax := math.Max(clr.R, math.Max(clr.G, clr.B))
		cMin := math.Min(clr.R, math.Min(clr.G, clr.B))
		delta := cMax - cMin

		var h, s, v float64

		// hue: common with HSL
		h = getHue6(clr, cMax, delta)

		// saturation: 0 if cMax is 0; otherwise, a division
		if cMax > 0.0 {
			s = delta / cMax
		}

		// value: maximum of any RGB channel
		v = cMax

		// quantize
		hF := limitQuantize(h+ce[0], 0.0, 6.0, levels[0])
		sF := normQuantize(s+ce[1], levels[1])
		vF := normQuantize(v+ce[2], levels[2])
		aF := normQuantize(clr.A+ce[3], levels[3])

		if !config.Flat {
			eVal := ErrorValue{h - hF, s - sF, v - vF, clr.A - aF}
			diffuseFloydSteinberg(&ctx.e, ctx.x, eVal)
		}

		// convert back to RGB (Wikipedia version)
		// C = chroma; X = intermediate (second-highest color value); M is to
		// match value.
		C := vF * sF
		X := C * (1 - math.Abs(modPositively(hF, 2.0)-1))
		M := vF - C

		// shared with HSL
		r, g, b := rgbFromHCXM(hF, C, X, M)

		return color.NRGBA{
			R: int8OfFloat(r),
			G: int8OfFloat(g),
			B: int8OfFloat(b),
			A: uint8(aF * FMUL8),
		}
	}
}

func quantizerHsl(bits []int, ctx *dither, config Config) PixelQuantizer {
	levels := levelsFromBits(bits, "HSL[A]")

	return func(clr FloatNRGBA) color.NRGBA {
		ce := ctx.e[0][ctx.x]

		// convert to space
		cMax := math.Max(clr.R, math.Max(clr.G, clr.B))
		cMin := math.Min(clr.R, math.Min(clr.G, clr.B))
		delta := cMax - cMin

		// because l1 look so similar, use t for "lighTness"
		var h, s, t float64

		h = getHue6(clr, cMax, delta)
		t = cMax - delta/2
		if t > 0.0 && t < 1.0 {
			s = (cMax - t) / math.Min(t, 1.0-t)
		}

		// quantize
		hF := limitQuantize(h+ce[0], 0.0, 6.0, levels[0])
		sF := normQuantize(s+ce[1], levels[1])
		tF := normQuantize(t+ce[2], levels[2])
		aF := normQuantize(clr.A+ce[3], levels[3])

		if !config.Flat {
			eVal := ErrorValue{h - hF, s - sF, t - tF, clr.A - aF}
			diffuseFloydSteinberg(&ctx.e, ctx.x, eVal)
		}

		// convert back to RGB
		C := sF * (1 - math.Abs(2*tF-1))
		X := C * (1 - math.Abs(modPositively(hF, 2)-1))
		M := tF - C/2

		r, g, b := rgbFromHCXM(hF, C, X, M)

		return color.NRGBA{
			R: int8OfFloat(r),
			G: int8OfFloat(g),
			B: int8OfFloat(b),
			A: uint8(aF * FMUL8),
		}
	}
}

func matrixQuantizer(toSpace, fromSpace matrix.Matrix, levels LevelsValue, ctx *dither, config Config) PixelQuantizer {
	return func(clr FloatNRGBA) color.NRGBA {
		rgbIn, err := matrix.Build(matrix.Builder{
			matrix.Row{clr.R},
			matrix.Row{clr.G},
			matrix.Row{clr.B},
		})
		if err != nil {
			panic(err)
		}

		spaceIn, err := toSpace.DotProduct(rgbIn)
		if err != nil {
			panic(err)
		}
		// use U, V, W for the channels, because we don't know their real names
		// and this won't conflict with any of R, G, B, A for RGBA variables
		u, v, w := spaceIn.At(0, 0), spaceIn.At(1, 0), spaceIn.At(2, 0)

		// quantize without range clamping, because we don't know that info;
		// also, be careful to support passthrough (levels[i]==0.0)
		ce := ctx.e[0][ctx.x]
		uF, vF, wF := u+ce[0], v+ce[1], w+ce[2]
		if levels[0] > 0.0 {
			uF = math.RoundToEven(uF*levels[0]) / levels[0]
		}
		if levels[1] > 0.0 {
			vF = math.RoundToEven(vF*levels[1]) / levels[1]
		}
		if levels[2] > 0.0 {
			wF = math.RoundToEven(wF*levels[2]) / levels[2]
		}
		aF := normQuantize(clr.A+ce[3], levels[3])

		if !config.Flat {
			eVal := ErrorValue{u - uF, v - vF, w - wF, clr.A - aF}
			diffuseFloydSteinberg(&ctx.e, ctx.x, eVal)
		}

		// convert back to RGB (build matrix, multiply, read)
		uvwOut, err := matrix.Build(matrix.Builder{
			matrix.Row{uF},
			matrix.Row{vF},
			matrix.Row{wF},
		})
		if err != nil {
			panic(err)
		}

		spaceOut, err := fromSpace.DotProduct(uvwOut)
		if err != nil {
			panic(err)
		}

		r, g, b := spaceOut.At(0, 0), spaceOut.At(1, 0), spaceOut.At(2, 0)

		return color.NRGBA{
			R: int8OfFloat(r),
			G: int8OfFloat(g),
			B: int8OfFloat(b),
			A: uint8(aF * FMUL8),
		}
	}
}

func quantizerYuv(bits []int, ctx *dither, config Config) PixelQuantizer {
	levels := levelsFromBits(bits, "YUV[A]")

	var toYuv, fromYuv matrix.Matrix

	// set up the conversion matrices
	if config.BT470 {
		toYuv, _ = matrix.Build(
			matrix.Builder{
				matrix.Row{0.299, 0.587, 0.114},
				matrix.Row{-0.014713, -0.28886, 0.436},
				matrix.Row{0.615, -0.51499, -1.0001},
			})
		fromYuv, _ = matrix.Build(
			matrix.Builder{
				matrix.Row{1, 0, 1.13983},
				matrix.Row{1, -0.39465, -0.58060},
				matrix.Row{1, 2.03211, 0},
			})
	} else {
		toYuv, _ = matrix.Build(
			matrix.Builder{
				matrix.Row{0.2126, 0.7152, 0.0722},
				matrix.Row{-0.09991, -0.33609, 0.436},
				matrix.Row{0.615, -0.55861, -0.05639},
			})
		fromYuv, _ = matrix.Build(
			matrix.Builder{
				matrix.Row{1, 0, 1.28033},
				matrix.Row{1, -0.21482, -0.38059},
				matrix.Row{1, 2.12798, 0},
			})
	}

	return matrixQuantizer(toYuv, fromYuv, levels, ctx, config)
}

func quantizerYiq(bits []int, ctx *dither, config Config) PixelQuantizer {
	levels := levelsFromBits(bits, "YIQ[A]")

	var toYiq, fromYiq matrix.Matrix

	// set up the conversion matrices
	if config.BT470 {
		toYiq, _ = matrix.Build(
			matrix.Builder{
				matrix.Row{0.299, 0.587, 0.114},
				matrix.Row{0.5959, -0.2746, -0.3213},
				matrix.Row{0.2115, -0.5227, 0.3112},
			})
		fromYiq, _ = matrix.Build(
			matrix.Builder{
				matrix.Row{1, 0.956, 0.619},
				matrix.Row{1, -0.272, -0.647},
				matrix.Row{1, -1.106, 1.703},
			})
	} else {
		// there is no HDTV YIQ, so this is the FCC NTSC standard instead.
		// at least this way, "-bt470" keeps meaning the same thing.
		toYiq, _ = matrix.Build(
			matrix.Builder{
				matrix.Row{0.30, 0.59, 0.11},
				matrix.Row{0.599, -0.2773, -0.3217},
				matrix.Row{0.213, -0.5251, 0.3121},
			})
		fromYiq, _ = matrix.Build(
			matrix.Builder{
				matrix.Row{1, 0.9469, 0.6236},
				matrix.Row{1, -0.2748, -0.6357},
				matrix.Row{1, -1.1, 1.7},
			})
	}

	return matrixQuantizer(toYiq, fromYiq, levels, ctx, config)
}

func parseBits(channels int, bits []int, from string, space string) []int {
	if len(from) != channels {
		log.Fatalf("Channel count mismatch: %d for %s color space", len(from), space)
	}

	for i := 0; i < channels; i += 1 {
		chr := from[i : i+1]
		if chr == "_" {
			// explicit pass-through: 0 bits => 0 levels => not quantized
			bits = append(bits, 0)
			continue
		}

		v, err := strconv.Atoi(chr)
		if err != nil {
			log.Fatalf("Channel at %d=\"%s\": %v", i, from[i:i+1], err)
		}

		bits = append(bits, v)
	}

	return bits
}

func resolveQuantizer(ctx *dither, config Config) PixelQuantizer {
	re := regexp.MustCompile(`^(?i)([a-z]+)([1-9_]+)$`)
	var builder QuantizeBuilder

	m := re.FindStringSubmatch(config.Space)
	if m == nil {
		log.Fatal("Could not parse colorspace: ", config.Space)
	}

	channels := 3
	space := strings.ToUpper(m[1])
	if len(space) == 4 || space == "YA" {
		space = space[:len(space)-1]
		channels += 1
	}
	switch space {
	case "Y":
		channels -= 2
		builder = quantizerGray
	case "RGB":
		builder = quantizerRgb
	case "HSV":
		builder = quantizerHsv
	case "HSL":
		builder = quantizerHsl
	case "YUV":
		builder = quantizerYuv
	case "YIQ":
		builder = quantizerYiq
	default:
		log.Fatal("Unknown color space: ", m[1])
	}

	bits := make([]int, 0, channels)
	bits = parseBits(channels, bits, m[2], m[1])
	return builder(bits, ctx, config)
}

func ProcessImage(m image.Image, qFunc PixelQuantizer, ctx *dither) *image.RGBA {
	bounds := m.Bounds()

	xMin := bounds.Min.X
	xMax := bounds.Max.X
	yMin := bounds.Min.Y
	yMax := bounds.Max.Y
	w := xMax - xMin
	h := bounds.Max.Y - yMin
	o := image.NewRGBA(image.Rect(0, 0, w, h))

	for i := range ctx.e {
		ctx.e[i] = make([]ErrorValue, w)
	}
	for y := yMin; y < yMax; y++ {
		// slide error diffusion window upward: copy(dst, src)
		copy(ctx.e[0], ctx.e[1])
		// fill newly-available window with 0
		for i := range ctx.e[1] {
			ctx.e[1][i] = ErrorValue{}
		}

		// compute output-relative Y coordinate
		oy := y - yMin
		for x := xMin; x < xMax; x++ {
			// compute output-relative X coordinate
			ox := x - xMin
			ctx.x = ox

			clr := straightAlphaFloat(m.At(x, y))
			o.Set(ox, oy, qFunc(clr))
		}
	}

	return o
}

// Process processes the image according to the configuration.
func Process(config Config) {
	// set up quantization color space and channel depths
	var dCtx dither
	qFunc := resolveQuantizer(&dCtx, config)

	// Decode the image data from a file
	reader, err := os.Open(config.SourceFile)
	if err != nil {
		log.Fatal(err)
	}
	m, _, err := image.Decode(reader)
	if err != nil {
		log.Fatal(err)
	}

	// Process (convert/quantize/maybe-dither/return) the image
	o := ProcessImage(m, qFunc, &dCtx)

	// Open the output
	writer, err := os.Create(config.OutputFile)
	defer writer.Close()

	// Encode the image to the output
	err = png.Encode(writer, o)
	if err != nil {
		log.Fatal("PNG output:", err)
	}
}

func main() {
	flag.Parse()
	Process(fConfig)
}
