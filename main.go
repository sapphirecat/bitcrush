package main

import (
	"flag"
	"image"
	"log"
	"math"
	"os"

	// Specific image-types are not used explicitly in the code below,
	// but are imported for its initialization side-effect, which allows
	// image.Decode to understand these image formats.
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
)

var SourceFile = flag.String("in", "img.png", "Source image to process")
var OutputFile = flag.String("out", "output.png", "Output image file name")

const FMUL = 65536.0

func straightAlphaFloat(c color.Color) (float32, float32, float32, float32) {
	r, g, b, a := c.RGBA()

	rF := float32(r) / FMUL
	gF := float32(g) / FMUL
	bF := float32(b) / FMUL
	aF := float32(a) / FMUL

	return rF / aF, gF / aF, bF / aF, aF
}

func diffuseFloydSteinberg(errorRows *[2][]float64, ox int, yerr float64) {
	errorRows[0][ox+1] += yerr * (7.0 / 16)
	if ox > 0 {
		errorRows[1][ox-1] += yerr * (3.0 / 16)
	}
	errorRows[1][ox] += yerr * (5.0 / 16)
	errorRows[1][ox+1] += yerr * (1.0 / 16)
}

func diffuseBitCrush(errorRows *[2][]float64, ox int, yerr float64) {
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
	var errorRows [2][]float64
	for i := range errorRows {
		// we are going to make the rows 1 wider, so writing always works
		errorRows[i] = make([]float64, 1+w)
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

			y0 := float64(0.299*r + 0.587*g + 0.114*b)
			// quantize, with error diffusion included
			yflr := math.Floor((y0+errorRows[0][ox])*16) / 16
			diffuseFloydSteinberg(&errorRows, ox, y0-yflr)
			//diffuseBitCrush(&errorRows, ox, y0-yflr)

			if yflr > 255.0/256 {
				yflr = 255.0 / 256
			} else if yflr < 0.0 {
				yflr = 0.0
			}
			yq := uint8(yflr * 256.0)
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
