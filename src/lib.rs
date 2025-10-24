#![doc = include_str!("../README.md")]

mod error;
mod telea;

#[cfg(feature = "image")]
use std::ops::Deref;

use error::Result;
use glam::USizeVec2;
use ndarray::Array2;
use num_traits::AsPrimitive;
pub use telea::telea_inpaint;

#[cfg(feature = "image")]
use image::{ImageBuffer, Luma, LumaA, Primitive, Rgb, Rgba};

#[cfg(feature = "image")]
/// Inpaint implementations for the `Image` crate.
pub trait Inpaint {
    fn telea_inpaint<P, MaskContainer>(
        &mut self,
        mask: &ImageBuffer<Luma<P>, MaskContainer>,
        radius: i32,
    ) -> Result<()>
    where
        P: Primitive,
        P: AsPrimitive<f32>,
        MaskContainer: Deref<Target = [P]>,
        f32: AsPrimitive<P>;
}

/// Create the implementation for a specific image type
///
/// Using macros as there is no stable way yet to
/// use inherited consts reliably without nightly features
macro_rules! create_inpaint_implementation {
    ($image_type:ty, $channels:expr, $format:ty) => {
        #[cfg(feature = "image")]
        impl Inpaint for $image_type {
            /// Inpaint image with provided mask using Telea algorithm.
            fn telea_inpaint<P, MaskContainer>(
                &mut self,
                mask: &ImageBuffer<Luma<P>, MaskContainer>,
                radius: i32,
            ) -> Result<()>
            where
                P: Primitive + 'static,
                P: AsPrimitive<f32>,
                MaskContainer: Deref<Target = [P]>,
                f32: AsPrimitive<P>,
            {
                let resolution = self.dimensions();
                let resolution = USizeVec2::new(resolution.0 as usize, resolution.1 as usize);
                let mut process_image: Array2<[$format; $channels]> = Array2::from_shape_vec(
                    (resolution.y, resolution.x),
                    unsafe {
                        std::slice::from_raw_parts(
                            self.as_raw().as_ptr() as *const [$format; $channels],
                            self.len() / $channels,
                        )
                    }
                    .to_vec(),
                )?;

                let mask: Array2<P> =
                    Array2::from_shape_vec((resolution.y, resolution.x), mask.as_raw().to_vec())?;

                telea_inpaint::<$format, $channels, P>(&mut process_image, mask, radius)?;

                let flattened: &[$format] = unsafe {
                    std::slice::from_raw_parts(process_image.as_ptr() as *const $format, self.len())
                };
                self.copy_from_slice(flattened);
                Ok(())
            }
        }
    };
}

macro_rules! create_inpaint_implementations {
    ($($pixel_type:ident, $type:ty, $channels:expr),*) => {
        $(
            #[cfg(feature = "image")]
            create_inpaint_implementation!(ImageBuffer<$pixel_type<$type>, Vec<$type>>, $channels, $type);
        )*
    };
}

create_inpaint_implementations! {
    Rgba, f32, 4,
    Rgb, f32, 3,
    LumaA, f32, 2,
    Luma, f32, 1,
    Rgba, i32, 4,
    Rgb, i32, 3,
    LumaA, i32, 2,
    Luma, i32, 1,
    Rgba, u32, 4,
    Rgb, u32, 3,
    LumaA, u32, 2,
    Luma, u32, 1,
    Rgba, u16, 4,
    Rgb, u16, 3,
    LumaA, u16, 2,
    Luma, u16, 1,
    Rgba, u8, 4,
    Rgb, u8, 3,
    LumaA, u8, 2,
    Luma, u8, 1
}

#[cfg(test)]
mod tests {
    use image::{DynamicImage, ImageBuffer, Luma, LumaA, Rgb, Rgba};
    use rstest::*;
    use std::path::PathBuf;

    use crate::Inpaint;

    macro_rules! create_inpaint_test_cases {
        ($pixel_type:ty, $precision:ty, $test_name:ident) => {
            paste::item! {
                #[rstest]
                #[case(
                PathBuf::from("./test/images/input/bird.png"),
                PathBuf::from("./test/images/mask/thin.png"),
                PathBuf::from(format!("./test/images/expected/{}/bird_thin.png", "telea"))
                )]
                #[case(
                    PathBuf::from("./test/images/input/bird.png"),
                    PathBuf::from("./test/images/mask/medium.png"),
                    PathBuf::from(format!("./test/images/expected/{}/bird_medium.png", "telea"))
                )]
                #[case(
                    PathBuf::from("./test/images/input/bird.png"),
                    PathBuf::from("./test/images/mask/large.png"),
                    PathBuf::from(format!("./test/images/expected/{}/bird_large.png", "telea"))
                )]
                #[case(
                    PathBuf::from("./test/images/input/bird.png"),
                    PathBuf::from("./test/images/mask/text.png"),
                    PathBuf::from(format!("./test/images/expected/{}/bird_text.png", "telea"))
                )]
                #[case(
                    PathBuf::from("./test/images/input/frog.png"),
                    PathBuf::from("./test/images/mask/thin.png"),
                    PathBuf::from(format!("./test/images/expected/{}/frog_thin.png", "telea"))
                )]
                #[case(
                    PathBuf::from("./test/images/input/frog.png"),
                    PathBuf::from("./test/images/mask/medium.png"),
                    PathBuf::from(format!("./test/images/expected/{}/frog_medium.png", "telea"))
                )]
                #[case(
                    PathBuf::from("./test/images/input/frog.png"),
                    PathBuf::from("./test/images/mask/text.png"),
                    PathBuf::from(format!("./test/images/expected/{}/frog_text.png", "telea"))
                )]
                fn [< test_inpaint_image_type _ $test_name >](
                    #[case] image: PathBuf,
                    #[case] mask: PathBuf,
                    #[case] expected: PathBuf,
                ) {
                    let mut image: ImageBuffer<$pixel_type<$precision>, Vec<$precision>> = image::open(image).unwrap().into();
                    let expected: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(expected).unwrap().into();
                    let mask = image::open(mask).unwrap().to_luma8();

                    image.telea_inpaint(&mask, 5).unwrap();
                    let comparison_score = image_compare::rgb_hybrid_compare(
                        &DynamicImage::from(image).to_rgb8(),
                        &expected,
                    )
                    .unwrap()
                    .score;
                    assert!(comparison_score >= 0.99);

                }
            }
        };
    }
    create_inpaint_test_cases!(Rgba, f32, rgbaf32);
    create_inpaint_test_cases!(Rgba, u16, rgbau16);
    create_inpaint_test_cases!(Rgb, u16, rgbu16);
    create_inpaint_test_cases!(LumaA, u16, lumaau16);
    create_inpaint_test_cases!(Luma, u16, lumau16);
    create_inpaint_test_cases!(Rgba, u8, rgbau8);
    create_inpaint_test_cases!(Rgb, u8, rgbu8);
    create_inpaint_test_cases!(LumaA, u8, lumaau8);
    create_inpaint_test_cases!(Luma, u8, lumau8);
}
