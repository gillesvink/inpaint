use inpaint::prelude::*;
use std::time::Instant;

fn main() {
    let mut image = image::open("../../test/images/baked/frog.png").unwrap().to_rgb8();
    let mask = image::open("../../test/images/mask/text.png").unwrap().to_luma8();

    let start_time = Instant::now();
    image.telea_inpaint(&mask, 5).unwrap();
    let elapsed_time = start_time.elapsed();

    image.save("./output.png").unwrap(); 

    println!("Inpainting finished in {:.2?} second(s).", elapsed_time);
}
