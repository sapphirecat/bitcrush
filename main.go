package main

import (
	"flag"
	"image"
	"log"
	"os"

	// Specific image-types are not used explicitly in the code below,
	// but are imported for its initialization side-effect, which allows
	// image.Decode to understand these image formats.
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"

	"github.com/goki/mat32"
)

var SourceFile = flag.String("in", "img.png", "Source image to process")
var OutputFile = flag.String("out", "output.png", "Output image file name")
var WithDiffusion = flag.Bool("dither", true, "FALSE to disable dithering")
var BT470 = flag.Bool("sd", false, "TRUE to enable SDTV BT.470 luma weights")

const FMUL = 65535.0

func straightAlphaFloat(c color.Color) (float32, float32, float32, float32) {
	r, g, b, a := c.RGBA()

	rF := float32(r) / FMUL
	gF := float32(g) / FMUL
	bF := float32(b) / FMUL
	aF := float32(a) / FMUL

	return rF / aF, gF / aF, bF / aF, aF
}

func diffuseFloydSteinberg(errorRows *[2][]float32, ox int, yerr float32) {
	errorRows[0][ox+1] += yerr * (7.0 / 16)
	if ox > 0 {
		errorRows[1][ox-1] += yerr * (3.0 / 16)
	}
	errorRows[1][ox] += yerr * (5.0 / 16)
	errorRows[1][ox+1] += yerr * (1.0 / 16)
}

func diffuseBitCrush(errorRows *[2][]float32, ox int, yerr float32) {
	errorRows[0][ox+1] += yerr * 0.5
	errorRows[1][ox] = yerr * 0.5
}

func main() {
	flag.Parse()

	// make sure this pointer is real
	if SourceFile == nil || OutputFile == nil {
		log.Fatal("flag.Parse() -> nil!?")
	}

	// Decode the image data from a file
	reader, err := os.Open(*SourceFile)
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
	var errorRows [2][]float32
	for i := range errorRows {
		// we are going to make the rows 1 wider, so writing always works
		errorRows[i] = make([]float32, 1+w)
	}
	for y := yMin; y < yMax; y++ {
		// slide error diffusion window upward: copy(dst, src)
		copy(errorRows[0], errorRows[1])
		// fill newly-available window with 0
		for i := range errorRows[1] {
			errorRows[1][i] = 0.0
		}

		for x := xMin; x < xMax; x++ {
			// A color's RGBA method returns values in the range [0, 65535],
			// with premultiplied alpha.  Convert to normalized float32 with
			// with straight alpha.
			r, g, b, a := straightAlphaFloat(m.At(x, y))

			ox := x - xMin
			oy := y - yMin

			var y0 float32
			if BT470 != nil && *BT470 {
				y0 = float32(0.299*r + 0.587*g + 0.114*b) // BT.470
			} else {
				y0 = float32(0.2126*r + 0.7152*g + 0.0722*b) // BT.709
			}
			// quantize, with error diffusion included
			yflr := mat32.RoundToEven((y0+errorRows[0][ox])*15) / 15
			if WithDiffusion == nil || *WithDiffusion {
				diffuseFloydSteinberg(&errorRows, ox, y0-yflr)
				//diffuseBitCrush(&errorRows, ox, y0-yflr)
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
	writer, err := os.Create(*OutputFile)
	defer writer.Close()

	// Encode the image to the output
	err = png.Encode(writer, o)
	if err != nil {
		log.Fatal("PNG output:", err)
	}
}
