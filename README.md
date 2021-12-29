# bitcrush

Color quantization experiment, version 4

## Build

    git clone https://github.com/sapphirecat/bitcrush
    cd bitcrush
    go build

Tested on go1.18beta1 linux/amd64, although there are no 1.18 features in use.

## Run

    ./bitcrush -in image.jpg -out image-332.png -space rgb332
    ./bitcrush -in image.jpg -out image-y4.png -space y4 -sd

Options:

- `-in` Input file, default `img.png`
- `-out` Output file, default `output.png`
- `-flat` Do not dither images
- `-space` Color space description (see below)
- `-sd` For Y* color spaces, use BT.470 luma weights, not BT.703 (or FCC NTSC)

## Color space descriptions

Color space descriptions are made up of an _identifier_ for the color space,
followed by the _bit depths_ to reduce each channel to.  Each color space may
optionally process alpha by appending "A" and including a bit depth for the
alpha channel.  "_" for a bit depth means to skip quantization for the channel
entirely; it will remain 8-bit.  "0" is not a valid bit depth.

Supported color spaces:

1. **Y:** black-and-white only, similar to desaturating by "luma" in a popular
  open-source image editor
2. **RGB:** Red/Green/Blue, the additive color primaries
3. **HSV:** Hue/Saturation/Value, where value expresses black to color
4. **HSL:** Hue/Saturation/Lightness, where lightness expresses black to color
  (at the midpoint) to white
5. **YUV:** PAL/SECAM TV, with UV representing blue-yellow and red-green
  differences
6. **YIQ:** NTSC (North America) TV, with IQ representing blue-orange and
  green-magenta differences

Thus, `Y4` means 16 levels of gray (2 raised to the 4th power.)  This is
identical to `YA4_`, which leaves the alpha channel unchanged.  In contrast,
`YA41` produces two levels of alpha (fully transparent and fully opaque.)

## Previously...

- [blit-crusher](https://github.com/sapphirecat/blit-crusher) in F#
- Python3/Pillow and Python2/PIL variants

## License

MIT.
