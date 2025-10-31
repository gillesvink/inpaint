[![Tests](https://ci.codeberg.org/api/badges/15490/status.svg)](https://ci.codeberg.org/repos/15490)
[![License](https://img.shields.io/crates/l/inpaint)](https://crates.io/crates/inpaint) 
[![Version](https://img.shields.io/crates/v/inpaint)](https://crates.io/crates/inpaint) 
[![PyPI - Downloads](https://img.shields.io/pypi/dm/inpaint)](https://pypi.org/project/inpaint/) 
[![Python Versions](https://img.shields.io/pypi/pyversions/inpaint)](https://pypi.org/project/inpaint/) 

# Inpaint

Inpaint crate for image restoration and smooth interpolation of unknown values.

While inpainting is used for Images, this crate exposes its interface with [ndarrays](https://docs.rs/ndarray/latest/ndarray/).
Unlike OpenCV, any channel count and pixel type can be used.

## Add to your project

For Rust, when you want to use it on images
```bash
cargo add inpaint --features image
```

Or in Python with uv 
```bash
uv add inpaint
```

## Information

The [Telea](https://codeberg.org/gillesvink/inpaint/src/branch/main/src/telea.rs) algorithm is ported from the [Pyheal](https://github.com/olvb/Pyheal) project, with some optimizations for Rust. With the Python bindings the same result can be achieved with this crate. In testing it is over 30x faster than Pyheal. The sample image takes `0.6` second in Pyheal, while in this crate it takes around `0.02` seconds on my machine.

Lets have this image of the toad I recently photographed. Unfortunately, some text has been burned into the image which I desperately want to remove:

| Damaged image           |  Mask                   |
|-------------------------|-------------------------|
| ![toad](https://codeberg.org/gillesvink/inpaint/media/branch/main/test/images/baked/toad.png) | ![Mask](https://codeberg.org/gillesvink/inpaint/media/branch/main/test/images/mask/text.png) |

---
Running this crate on the image returns this as the result:

| Result                  |
|-------------------------|
| ![Result](https://codeberg.org/gillesvink/inpaint/media/branch/main/test/images/expected/telea/toad_text.png) |

You can call this code yourself at `./examples/python/` or `./examples/rust/`.
```bash
cd examples/python && uv run main.py
```

```bash
cd examples/rust && cargo run --release
```




## Features
- Non-image support, so any array can be used as long as it is in the `ndarray` format.
- Traits for the `Image` crate as optional feature. Just call `.inpaint_telea()` method on your image and have it inpainted. Make sure the `image` feature is enabled in your `Cargo.toml`
- Python bindings to have the same functionality as Rust in Python.

## Examples

### Inpaint an ImageBuffer in Rust
You can also run the example in examples/simple. This will use the inpaint library and output the inpainted image as `output.png`.

> [!IMPORTANT]  
> You need to have the `image` feature enabled.

```rust
use inpaint::prelude::*;
let mut image = image::open("./test/images/input/toad.png").unwrap().to_rgba32f();
let mask = image::open("./test/images/mask/text.png").unwrap().to_luma32f();

#[cfg(feature = "image")] // feature needs to be enabled for it to work
image.telea_inpaint(&mask, 3);
```

### Inpaint an image in Python
```python
import inpaint
from PIL import Image

image = Image.open("./test/images/input/toad.png")
mask = Image.open("./test/images/mask/text.png")

output = inpaint.telea(image, mask, 3)

output.save("./output.png")
```

### Inpaint an array in Rust

When not using the Image crate, just use the raw ndarrays.

```rust
use inpaint::telea_inpaint;
use ndarray::{Array2, Array3};
use glam::USizeVec2;

let resolution = USizeVec2::new(1920, 1080);
let channels = 4;
// obviously you need to use actual data, this is just an example
let mut input_image = Array3::from_elem((resolution.y, resolution.x, channels), 0.0);
let mask = Array2::from_elem((resolution.y, resolution.x), 0.0);

telea_inpaint(&mut input_image.view_mut(), &mask.view(), 1).unwrap();
```