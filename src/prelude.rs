#[cfg(feature = "image")]
mod image {
    use image::{ImageBuffer, Luma, Pixel, Primitive};
    
    use std::ops::Deref;
    
    use crate::error::Result;
    use glam::USizeVec2;
    
    use ndarray::{Array2, Array3};
    use num_traits::AsPrimitive;
    
    /// Inpaint implementations for the `Image` crate.
    pub trait ImageInpaint {
        /// Inpaint image with provided mask using Telea algorithm.
        fn telea_inpaint<MaskPixel, MaskContainer>(
            &mut self,
            mask: &ImageBuffer<Luma<MaskPixel>, MaskContainer>,
            radius: i32,
        ) -> Result<()>
        where
            MaskPixel: Primitive + AsPrimitive<f32> + 'static,
            MaskContainer: Deref<Target = [MaskPixel]>;
    }
    
    #[cfg(feature = "image")]
    impl<ImagePixel, ImageContainer> ImageInpaint for ImageBuffer<ImagePixel, Vec<ImageContainer>>
    where
        ImagePixel: Pixel<Subpixel = ImageContainer>,
        ImageContainer: Clone + Copy + AsPrimitive<f32>,
        f32: AsPrimitive<ImageContainer>,
    {
        fn telea_inpaint<MaskPixel, MaskContainer>(
            &mut self,
            mask: &ImageBuffer<Luma<MaskPixel>, MaskContainer>,
            radius: i32,
        ) -> Result<()>
        where
            MaskPixel: Primitive + AsPrimitive<f32> + 'static,
            MaskContainer: Deref<Target = [MaskPixel]>,
        {
            use crate::telea_inpaint;
    
            let resolution = self.dimensions();
            let resolution = USizeVec2::new(resolution.0 as usize, resolution.1 as usize);
    
            let mut process_image: Array3<ImageContainer> = Array3::from_shape_vec(
                (
                    resolution.y,
                    resolution.x,
                    ImagePixel::CHANNEL_COUNT as usize,
                ),
                self.as_raw().to_vec(),
            )?;
    
            let mask: Array2<MaskPixel> =
                Array2::from_shape_vec((resolution.y, resolution.x), mask.as_raw().to_vec())?;
    
            telea_inpaint(&mut process_image, mask, radius)?;
    
            self.copy_from_slice(process_image.as_slice().unwrap());
            Ok(())
        }
    }
    
    
    #[cfg(test)]
    mod tests {
        use image::{DynamicImage, ImageBuffer, Luma, LumaA, Rgb, Rgba};
        use rstest::*;
        use std::path::PathBuf;
        use super::*;
    
        
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
                        PathBuf::from("./test/images/input/toad.png"),
                        PathBuf::from("./test/images/mask/thin.png"),
                        PathBuf::from(format!("./test/images/expected/{}/toad_thin.png", "telea"))
                    )]
                    #[case(
                        PathBuf::from("./test/images/input/toad.png"),
                        PathBuf::from("./test/images/mask/medium.png"),
                        PathBuf::from(format!("./test/images/expected/{}/toad_medium.png", "telea"))
                    )]
                    #[case(
                        PathBuf::from("./test/images/input/toad.png"),
                        PathBuf::from("./test/images/mask/text.png"),
                        PathBuf::from(format!("./test/images/expected/{}/toad_text.png", "telea"))
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
                            &DynamicImage::from(image.clone()).to_rgb8(),
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
}


#[cfg(feature = "image")]
pub use image::*;