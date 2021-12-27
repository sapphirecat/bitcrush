package main

import (
	"flag"
	"image"
	"log"
	"os"

	// Specific image-types are not used explicitly in the code below,
	// but importing registers them for use with image.Decode.
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"

	"github.com/goki/mat32"
)

type Config struct {
	SourceFile, OutputFile string
	Space                  string
	Flat, BT470            bool
}

type PixelQuantizer func(FloatRGBA) color.RGBA
type QuantizeBuilder func([]int, *dither, Config) PixelQuantizer
type ErrorValue [3]float32
type Errors [2][]ErrorValue
type FloatRGBA struct {
	R, G, B, A float32
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

// straightAlphaFloat turns a color.Color into normalized float32 components
// with straight (not premultiplied) alpha.
func straightAlphaFloat(c color.Color) FloatRGBA {
	r, g, b, a := c.RGBA()
	aDiv := float32(a)

	// for the RGB components, FMUL cancels out: (r/FMUL) / (a/FMUL) = r/a
	return FloatRGBA{
		R: float32(r) / aDiv,
		G: float32(g) / aDiv,
		B: float32(b) / aDiv,
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

func levelsForBits(bits int) int {
	if bits < 0 {
		panic("negative bits requested")
	}

	return (1 << bits) - 1
}

func int8OfFloat(value float32) uint8 {
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

	levels := float32(levelsForBits(bits[0]))

	return func(clr FloatRGBA) color.RGBA {
		// compute the full-precision grayscale
		var y0 float32
		if config.BT470 {
			y0 = float32(0.299*clr.R + 0.587*clr.G + 0.114*clr.B) // BT.470
		} else {
			y0 = float32(0.2126*clr.R + 0.7152*clr.G + 0.0722*clr.B) // BT.709
		}

		// quantize
		yflr := mat32.RoundToEven((y0+ctx.e[0][ctx.x][0])*levels) / levels
		if !config.Flat {
			diffuseFloydSteinberg(&ctx.e, ctx.x, ErrorValue{y0 - yflr})
		}

		// convert to output space
		yq := int8OfFloat(yflr)
		return color.RGBA{R: yq, G: yq, B: yq, A: uint8(clr.A * FMUL8)}
	}
}

func quantizerRgb(bits []int, ctx *dither, config Config) PixelQuantizer {
	var levels [3]float32

	if len(bits) < 3 {
		panic("not enough bits for RGB channels")
	}

	for i := range levels {
		levels[i] = float32(levelsForBits(bits[i]))
	}

	return func(clr FloatRGBA) color.RGBA {
		x := ctx.x

		rF := mat32.RoundToEven((clr.R+ctx.e[0][x][0])*levels[0]) / levels[0]
		gF := mat32.RoundToEven((clr.G+ctx.e[0][x][1])*levels[1]) / levels[1]
		bF := mat32.RoundToEven((clr.B+ctx.e[0][x][2])*levels[2]) / levels[2]

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

// Process processes the image according to the configuration.
func Process(config Config) {
	// Decode the image data from a file
	reader, err := os.Open(config.SourceFile)
	if err != nil {
		log.Fatal(err)
	}
	m, _, err := image.Decode(reader)
	if err != nil {
		log.Fatal(err)
	}
	bounds := m.Bounds()

	xMin := bounds.Min.X
	xMax := bounds.Max.X
	yMin := bounds.Min.Y
	yMax := bounds.Max.Y
	w := xMax - xMin
	h := bounds.Max.Y - yMin
	o := image.NewRGBA(image.Rect(0, 0, w, h))
	var dCtx dither
	var qFunc PixelQuantizer

	//qFunc = quantizerGray([]int{4}, &dCtx, config)
	qFunc = quantizerRgb([]int{3, 3, 2}, &dCtx, config)

	for i := range dCtx.e {
		dCtx.e[i] = make([]ErrorValue, w)
	}
	for y := yMin; y < yMax; y++ {
		// slide error diffusion window upward: copy(dst, src)
		copy(dCtx.e[0], dCtx.e[1])
		// fill newly-available window with 0
		for i := range dCtx.e[1] {
			dCtx.e[1][i] = ErrorValue{}
		}

		// compute output-relative Y coordinate
		oy := y - yMin
		for x := xMin; x < xMax; x++ {
			// compute output-relative X coordinate
			ox := x - xMin
			dCtx.x = ox

			clr := straightAlphaFloat(m.At(x, y))
			o.Set(ox, oy, qFunc(clr))
		}
	}

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
