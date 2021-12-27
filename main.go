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
	Flat, BT470            bool
}

type ErrorValue [3]float32
type Errors [2][]ErrorValue

var config = Config{}

func init() {
	flag.StringVar(&config.SourceFile, "in", "img.png", "Source image to process")
	flag.StringVar(&config.OutputFile, "out", "output.png", "Output image file name")
	flag.BoolVar(&config.Flat, "flat", false, "Skip dithering")
	flag.BoolVar(&config.BT470, "sd", false, "Use SDTV BT.470 luma weights instead of BT.703")
}

// FMUL is the factor between color.RGBA and normalized floats
const FMUL = 65535.0

// straightAlphaFloat turns a color.Color into normalized float32 components
// with straight (not premultiplied) alpha.
func straightAlphaFloat(c color.Color) (float32, float32, float32, float32) {
	r, g, b, a := c.RGBA()

	rF := float32(r) / FMUL
	gF := float32(g) / FMUL
	bF := float32(b) / FMUL
	aF := float32(a) / FMUL

	return rF / aF, gF / aF, bF / aF, aF
}

// diffuseFloydSteinberg implements the error diffusion algorithm on a window
// of rows.
func diffuseFloydSteinberg(errorRows *Errors, ox int, errVec ErrorValue) {
	for i, v := range errVec {
		if ox > 0 {
			errorRows[1][ox-1][i] += v * (3.0 / 16)
		}
		errorRows[1][ox][i] += v * (5.0 / 16)
		if ox < len(errorRows[0])-1 {
			errorRows[0][ox+1][i] += v * (7.0 / 16)
			errorRows[1][ox+1][i] += v * (1.0 / 16)
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
	var errorRows Errors
	for i := range errorRows {
		errorRows[i] = make([]ErrorValue, w)
	}
	for y := yMin; y < yMax; y++ {
		// slide error diffusion window upward: copy(dst, src)
		copy(errorRows[0], errorRows[1])
		// fill newly-available window with 0
		for i := range errorRows[1] {
			errorRows[1][i] = ErrorValue{}
		}

		for x := xMin; x < xMax; x++ {
			// A color's RGBA method returns values in the range [0, 65535],
			// with premultiplied alpha.  Convert to normalized float32 with
			// with straight alpha.
			r, g, b, a := straightAlphaFloat(m.At(x, y))

			ox := x - xMin
			oy := y - yMin

			var y0 float32
			if config.BT470 {
				y0 = float32(0.299*r + 0.587*g + 0.114*b) // BT.470
			} else {
				y0 = float32(0.2126*r + 0.7152*g + 0.0722*b) // BT.709
			}
			// quantize, with error diffusion included
			yflr := mat32.RoundToEven((y0+errorRows[0][ox][0])*15) / 15
			delta := y0 - yflr
			if !config.Flat {
				diffuseFloydSteinberg(&errorRows, ox, ErrorValue{delta, delta, delta})
			}

			if yflr > 1.0 {
				yflr = 1.0
			} else if yflr < 0.0 {
				yflr = 0.0
			}
			yq := uint8(yflr * 255.0)
			o.Set(ox, oy, color.RGBA{yq, yq, yq, uint8(a * FMUL)})
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
	Process(config)
}
