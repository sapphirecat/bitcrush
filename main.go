package main

import (
	"flag"
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

func levelsForBits(bits int) float64 {
	if bits < 0 {
		panic("negative bits requested")
	}

	levels := (1 << bits) - 1
	return float64(levels)
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

	levels := levelsForBits(bits[0])

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
	var levels [3]float64

	if len(bits) < 3 {
		panic("not enough bits for RGB channels")
	}

	for i := range levels {
		levels[i] = levelsForBits(bits[i])
	}

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

func quantizerHsv(bits []int, ctx *dither, config Config) PixelQuantizer {
	var levels [3]float64

	if len(bits) < 3 {
		panic("not enough bits for HSV channels")
	}

	for i := range levels {
		levels[i] = levelsForBits(bits[i])
	}

	return func(clr FloatRGBA) color.RGBA {
		x := ctx.x

		// convert to HSV
		cMax := math.Max(clr.R, math.Max(clr.G, clr.B))
		cMin := math.Min(clr.R, math.Min(clr.G, clr.B))
		delta := cMax - cMin

		var h, s, v float64

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

		var rP, gP, bP float64
		if hF < 1.0 {
			rP, gP = C, X
		} else if hF < 2.0 {
			rP, gP = X, C
		} else if hF < 3.0 {
			gP, bP = C, X
		} else if hF < 4.0 {
			gP, bP = X, C
		} else if hF < 5.0 {
			bP, rP = C, X
		} else {
			bP, rP = X, C
		}

		return color.RGBA{
			R: int8OfFloat(rP + M),
			G: int8OfFloat(gP + M),
			B: int8OfFloat(bP + M),
			A: uint8(clr.A * FMUL8),
		}
	}
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
