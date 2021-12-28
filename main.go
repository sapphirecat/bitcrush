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

	"github.com/oelmekki/matrix"
)

type Config struct {
	SourceFile, OutputFile string
	Space                  string
	Flat, BT470            bool
}

type PixelQuantizer func(FloatRGBA) color.RGBA
type QuantizeBuilder func([]int, *dither, Config) PixelQuantizer
type ErrorValue [3]float64
type Errors [2][]ErrorValue
type FloatRGBA struct {
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

// FMUL is the factor between color.RGBA and normalized floats
const FMUL = 65535.0

// FMUL8 is the factor between normalized floats and 8-bit representations
const FMUL8 = 255.0

// straightAlphaFloat turns a color.Color into normalized float64 components
// with straight (not premultiplied) alpha.
func straightAlphaFloat(c color.Color) FloatRGBA {
	r, g, b, a := c.RGBA()
	aDiv := float64(a)

	// for the RGB components, FMUL cancels out: (r/FMUL) / (a/FMUL) = r/a
	return FloatRGBA{
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
		if v == 0.0 {
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

func levelsFromBits(bits int) float64 {
	if bits < 0 {
		panic("negative bits requested")
	}

	levels := (1 << bits) - 1
	return float64(levels)
}

func triLevelsFromBits(bits []int, space string) (levels [3]float64) {
	if len(bits) < 3 {
		what := fmt.Sprintf("Not enough bits for %s channels (needed 3, got %d", space, len(bits))
		panic(what)
	}

	for i := range levels {
		levels[i] = levelsFromBits(bits[i])
	}

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

	levels := levelsFromBits(bits[0])

	return func(clr FloatRGBA) color.RGBA {
		// compute the full-precision grayscale
		var y0 float64
		if config.BT470 {
			y0 = float64(0.299*clr.R + 0.587*clr.G + 0.114*clr.B) // BT.470
		} else {
			y0 = float64(0.2126*clr.R + 0.7152*clr.G + 0.0722*clr.B) // BT.709
		}

		// quantize
		yflr := math.RoundToEven((y0+ctx.e[0][ctx.x][0])*levels) / levels
		if !config.Flat {
			diffuseFloydSteinberg(&ctx.e, ctx.x, ErrorValue{y0 - yflr})
		}

		// convert to output space
		yq := int8OfFloat(yflr)
		return color.RGBA{R: yq, G: yq, B: yq, A: uint8(clr.A * FMUL8)}
	}
}

func quantizerRgb(bits []int, ctx *dither, config Config) PixelQuantizer {
	levels := triLevelsFromBits(bits, "RGB")

	return func(clr FloatRGBA) color.RGBA {
		x := ctx.x

		rF := math.RoundToEven((clr.R+ctx.e[0][x][0])*levels[0]) / levels[0]
		gF := math.RoundToEven((clr.G+ctx.e[0][x][1])*levels[1]) / levels[1]
		bF := math.RoundToEven((clr.B+ctx.e[0][x][2])*levels[2]) / levels[2]

		if !config.Flat {
			eVal := ErrorValue{clr.R - rF, clr.G - gF, clr.B - bF}
			diffuseFloydSteinberg(&ctx.e, ctx.x, eVal)
		}

		return color.RGBA{
			R: int8OfFloat(rF),
			G: int8OfFloat(gF),
			B: int8OfFloat(bF),
			A: uint8(clr.A * FMUL8),
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

func getHue6(clr FloatRGBA, cMax, delta float64) (h float64) {
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
	levels := triLevelsFromBits(bits, "HSV")

	return func(clr FloatRGBA) color.RGBA {
		x := ctx.x

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
		hF := math.RoundToEven((h+ctx.e[0][x][0])*levels[0]) / levels[0]
		sF := math.RoundToEven((s+ctx.e[0][x][1])*levels[1]) / levels[1]
		vF := math.RoundToEven((v+ctx.e[0][x][2])*levels[2]) / levels[2]

		// clamp to range
		hF = math.Min(6.0, math.Max(hF, 0.0))
		sF = math.Min(1.0, math.Max(sF, 0.0))
		vF = math.Min(1.0, math.Max(vF, 0.0))

		if !config.Flat {
			eVal := ErrorValue{h - hF, s - sF, v - vF}
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

		return color.RGBA{
			R: int8OfFloat(r),
			G: int8OfFloat(g),
			B: int8OfFloat(b),
			A: uint8(clr.A * FMUL8),
		}
	}
}

func quantizerHsl(bits []int, ctx *dither, config Config) PixelQuantizer {
	levels := triLevelsFromBits(bits, "HSL")

	return func(clr FloatRGBA) color.RGBA {
		x := ctx.x

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
		hF := math.RoundToEven((h+ctx.e[0][x][0])*levels[0]) / levels[0]
		sF := math.RoundToEven((s+ctx.e[0][x][1])*levels[1]) / levels[1]
		tF := math.RoundToEven((t+ctx.e[0][x][2])*levels[2]) / levels[2]

		// clamp to range
		hF = math.Min(6.0, math.Max(hF, 0.0))
		sF = math.Min(1.0, math.Max(sF, 0.0))
		tF = math.Min(1.0, math.Max(tF, 0.0))

		if !config.Flat {
			eVal := ErrorValue{h - hF, s - sF, t - tF}
			diffuseFloydSteinberg(&ctx.e, ctx.x, eVal)
		}

		// convert back to RGB
		C := sF * (1 - math.Abs(2*tF-1))
		X := C * (1 - math.Abs(modPositively(hF, 2)-1))
		M := tF - C/2

		r, g, b := rgbFromHCXM(hF, C, X, M)

		return color.RGBA{
			R: int8OfFloat(r),
			G: int8OfFloat(g),
			B: int8OfFloat(b),
			A: uint8(clr.A * FMUL8),
		}
	}
}

func matrixQuantizer(toSpace, fromSpace matrix.Matrix, levels [3]float64, ctx *dither, config Config) PixelQuantizer {
	return func(clr FloatRGBA) color.RGBA {
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
		// and this won't conflict with any of R, G, B for RGB variables
		u, v, w := spaceIn.At(0, 0), spaceIn.At(1, 0), spaceIn.At(2, 0)

		// quantize
		x := ctx.x
		uF := math.RoundToEven((u+ctx.e[0][x][0])*levels[0]) / levels[0]
		vF := math.RoundToEven((v+ctx.e[0][x][1])*levels[1]) / levels[1]
		wF := math.RoundToEven((w+ctx.e[0][x][2])*levels[2]) / levels[2]

		if !config.Flat {
			eVal := ErrorValue{u - uF, v - vF, w - wF}
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

		return color.RGBA{
			R: int8OfFloat(r),
			G: int8OfFloat(g),
			B: int8OfFloat(b),
			A: uint8(clr.A * FMUL8),
		}
	}
}

func quantizerYuv(bits []int, ctx *dither, config Config) PixelQuantizer {
	levels := triLevelsFromBits(bits, "YUV")

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
	levels := triLevelsFromBits(bits, "YIQ")

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

func parseBits(channels int, bits []int, from string) []int {
	if len(from) < channels {
		log.Fatal("Not enough channels specified in: ", from)
	} else if len(from) > channels {
		log.Fatal("Too many channels specified in: ", from)
	}

	for i := 0; i < channels; i += 1 {
		v, err := strconv.Atoi(from[i : i+1])
		if err != nil {
			log.Fatalf("Channel at %d could not be parsed: %s", i, from[i:i+1])
		}

		bits = append(bits, v)
	}

	return bits
}

func resolveQuantizer(ctx *dither, config Config) PixelQuantizer {
	re := regexp.MustCompile(`^(?i)([a-z]+)([1-9]+)$`)
	var builder QuantizeBuilder

	m := re.FindStringSubmatch(config.Space)
	if m == nil {
		log.Fatal("Could not parse colorspace: ", config.Space)
	}

	channels := 3
	bits := make([]int, 0, 3)
	switch strings.ToUpper(m[1]) {
	case "Y":
		channels = 1
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

	bits = parseBits(channels, bits, m[2])
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
