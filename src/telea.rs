/// This is a Rust native port from the great code:
/// https://github.com/olvb/pyheal/blob/master/pyheal.py
///
/// And the research paper linked below.
///
/// I've not come up with this magic by myself in any way :)
/// There's a few changes in logic and Rust optimizations though.
///
/// Implementation details about telea's algorithm can be found at
/// https://www.olivier-augereau.com/docs/2004JGraphToolsTelea.pdf and
/// https://webspace.science.uu.nl/~telea001/Shapes/Inpainting
use crate::error::{Error, Result};
use core::cmp::Ordering;
use core::cmp::Reverse;
use core::f32;
use glam::{IVec2, USizeVec2, Vec2, Vec4};
use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayViewMut3, arr1, s};
use num_traits::AsPrimitive;
#[cfg(not(feature = "libm"))]
use num_traits::Float;
#[cfg(feature = "std")]
use std::collections::BinaryHeap;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::{collections::BinaryHeap, vec, vec::Vec};

/// Just a simple alias to the Array type
type Image<P> = Array3<P>;
/// Array containing pixel state flags
type FlagArray = Array2<Flag>;
/// Array containing distance to mask
type DistanceArray = Array2<f32>;

/// Max value as described in paper
const MAX: f32 = 1.0e6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Flags used to define a pixel state.
enum Flag {
    /// Pixel is outside boundary
    Known,
    /// Pixel that belongs to the narrow band
    Band,
    /// Pixel is inside boundary
    Inside,
}

impl Flag {
    /// Flip known to inside and inside to known
    pub fn flip(&self) -> Self {
        match self {
            Self::Known => Self::Inside,
            Self::Inside => Self::Known,
            _ => *self,
        }
    }
    /// Initialize flag from input bit
    pub fn from_value(value: u8) -> Self {
        match value {
            1 => Self::Band,
            _ => Self::Known,
        }
    }
}

#[derive(Debug, Clone)]
/// Item for in the NarrowBand.
///
/// It has a priority assigned which is the most important.
/// After that the y value is used for weight and then the x value.
struct QueueItem {
    pub priority: f32,
    pub coordinates: USizeVec2,
}

impl QueueItem {
    /// Initialize item from
    pub fn new(cost: f32, coordinates: USizeVec2) -> Self {
        Self {
            priority: cost,
            coordinates,
        }
    }
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        let cost_ordering = self
            .priority
            .partial_cmp(&other.priority)
            .unwrap_or(Ordering::Equal);

        match cost_ordering {
            Ordering::Equal => match self.coordinates.y.cmp(&other.coordinates.y) {
                Ordering::Equal => self.coordinates.x.cmp(&other.coordinates.x),
                ordering => ordering,
            },
            _ => cost_ordering,
        }
    }
}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for QueueItem {}

#[cfg(not(feature = "libm"))]
fn powi(value: f32, pow: i32) -> f32 {
    value.powi(pow)
}

#[cfg(feature = "libm")]
fn powi(value: f32, pow: i32) -> f32 {
    libm::powf(value, pow as f32)
}

#[cfg(not(feature = "libm"))]
fn sqrt(value: f32) -> f32 {
    value.sqrt()
}

#[cfg(feature = "libm")]
fn sqrt(value: f32) -> f32 {
    libm::sqrtf(value)
}
/// Solve the eikonal equation
fn solve_eikonal(
    a: IVec2,
    b: IVec2,
    resolution: USizeVec2,
    distances: &DistanceArray,
    flags: &FlagArray,
) -> f32 {
    if a.x < 0
        || a.y < 0
        || b.x < 0
        || b.y < 0
        || a.x >= resolution.x as i32
        || a.y >= resolution.y as i32
        || b.x >= resolution.x as i32
        || b.y >= resolution.y as i32
    {
        return MAX;
    };

    let a_usize = a.as_usizevec2();
    let b_usize = b.as_usizevec2();
    let a_flags = flags[[a_usize.y, a_usize.x]];
    let b_flags = flags[[b_usize.y, b_usize.x]];
    let a_distance = distances[[a_usize.y, a_usize.x]];
    let b_distance = distances[[b_usize.y, b_usize.x]];

    if a_flags == Flag::Known && b_flags == Flag::Known {
        let distance = 2.0 - powi(a_distance - b_distance, 2);
        if distance > 0.0 {
            let r = sqrt(distance);
            let mut s = (a_distance + b_distance - r) / 2.0;
            if s >= a_distance && s >= b_distance {
                return s;
            };
            s += r;
            if s >= a_distance && s >= b_distance {
                return s;
            }
            return MAX;
        }
    };

    if a_flags == Flag::Known {
        return 1.0 + a_distance;
    }
    if b_flags == Flag::Known {
        return 1.0 + b_distance;
    }
    MAX
}

/// Compute gradient weighting for both x and y
fn pixel_gradient(
    coordinates: USizeVec2,
    resolution: USizeVec2,
    distances: &DistanceArray,
    flags: &FlagArray,
) -> Vec2 {
    let distance = distances[[coordinates.y, coordinates.x]];

    let y;
    let next_y = coordinates.y + 1;
    if next_y >= resolution.y || coordinates.y == 0 {
        y = MAX;
    } else {
        let previous_y = coordinates.y - 1;

        let flag_previous = flags[[previous_y, coordinates.x]];
        let flag_next = flags[[next_y, coordinates.x]];

        if flag_previous != Flag::Inside && flag_next != Flag::Inside {
            y = (distances[[next_y, coordinates.x]] - distances[[previous_y, coordinates.x]]) / 2.0;
        } else if flag_previous != Flag::Inside {
            y = distance - distances[[previous_y, coordinates.x]];
        } else if flag_next != Flag::Inside {
            y = distances[[next_y, coordinates.x]] - distance;
        } else {
            y = 0.0;
        }
    }

    let x;
    let next_x = coordinates.x + 1;
    if next_x >= resolution.x || coordinates.x == 0 {
        x = MAX;
    } else {
        let previous_x = coordinates.x - 1;

        let flag_previous = flags[[coordinates.y, previous_x]];
        let flag_next = flags[[coordinates.y, next_x]];

        if flag_previous != Flag::Inside && flag_next != Flag::Inside {
            x = (distances[[coordinates.y, next_x]] - distances[[coordinates.y, previous_x]]) / 2.0;
        } else if flag_previous != Flag::Inside {
            x = distance - distances[[coordinates.y, previous_x]];
        } else if flag_next != Flag::Inside {
            x = distances[[coordinates.y, next_x]] - distance;
        } else {
            x = 0.0;
        }
    }

    Vec2::new(x, y)
}

/// Normalize value to 0-1 range in float
fn normalize_value<P>(value: P) -> f32
where
    P: AsPrimitive<f32>,
{
    value.as_()
        / match core::any::TypeId::of::<P>() {
            id if id == core::any::TypeId::of::<u8>() => u8::MAX as f32,
            id if id == core::any::TypeId::of::<u16>() => u16::MAX as f32,
            id if id == core::any::TypeId::of::<u32>() => u32::MAX as f32,
            id if id == core::any::TypeId::of::<u32>() => u64::MAX as f32,
            id if id == core::any::TypeId::of::<u32>() => u128::MAX as f32,
            id if id == core::any::TypeId::of::<i8>() => i8::MAX as f32,
            id if id == core::any::TypeId::of::<i16>() => i16::MAX as f32,
            id if id == core::any::TypeId::of::<i32>() => i32::MAX as f32,
            id if id == core::any::TypeId::of::<i32>() => i64::MAX as f32,
            id if id == core::any::TypeId::of::<i32>() => i128::MAX as f32,
            _ => 1.0,
        }
}

/// Convert the input array of any type to the FlagArray (which consists of enum values)
fn convert_mask_to_flag_array<P>(mask: &ArrayView2<P>, resolution: USizeVec2) -> FlagArray
where
    P: AsPrimitive<f32>,
{
    FlagArray::from_shape_fn((resolution.y, resolution.x), |(y, x)| {
        let value: f32 = normalize_value(mask[[y, x]]).ceil();
        Flag::from_value(value as u8)
    })
}

/// Get the coordinates around the specified coordinate
fn get_neighbors(coordinates: IVec2) -> [IVec2; 4] {
    [
        coordinates + IVec2::new(0, -1),
        coordinates + IVec2::new(-1, 0),
        coordinates + IVec2::new(0, 1),
        coordinates + IVec2::new(1, 0),
    ]
}

/// Calculate the distances between mask edges and pixels outside of mask area
fn compute_outside_distances(
    resolution: USizeVec2,
    distances: &mut DistanceArray,
    flags: &FlagArray,
    heap: &BinaryHeap<Reverse<QueueItem>>,
    radius: i32,
) -> Result<()> {
    let mut inner_flags = flags.clone().mapv(|f| f.flip());
    let mut current_heap = heap.clone();

    let mut last_distance = 0.0;
    let double_radius = radius as f32 * 2.0;
    while !current_heap.is_empty() {
        if last_distance >= double_radius {
            break;
        };

        let coordinates = if let Some(node) = current_heap.pop() {
            node.0.coordinates
        } else {
            break;
        };
        inner_flags[[coordinates.y, coordinates.x]] = Flag::Known;

        let neighbors = get_neighbors(coordinates.as_ivec2());
        for neighbor in neighbors {
            last_distance = match get_eikonal(resolution, distances, &mut inner_flags, neighbor) {
                Some(value) => value,
                None => continue,
            };
            distances[[neighbor.y as usize, neighbor.x as usize]] = last_distance;
            inner_flags[[neighbor.y as usize, neighbor.x as usize]] = Flag::Band;
            current_heap.push(Reverse(QueueItem::new(
                last_distance,
                neighbor.as_usizevec2(),
            )));
        }
    }
    *distances *= -1.0;
    Ok(())
}

/// Solve the eikonal equations to find the distance to the boundary
fn get_eikonal(
    resolution: USizeVec2,
    distances: &mut DistanceArray,
    flags: &mut FlagArray,
    neighbor: IVec2,
) -> Option<f32> {
    if neighbor.y < 0
        || neighbor.y >= resolution.y as i32
        || neighbor.x < 0
        || neighbor.x >= resolution.x as i32
    {
        return None;
    }
    if flags[[neighbor.y as usize, neighbor.x as usize]] != Flag::Inside {
        return None;
    }
    let eikonals = vec![
        solve_eikonal(
            neighbor + IVec2::new(0, -1),
            neighbor + IVec2::new(-1, 0),
            resolution,
            distances,
            flags,
        ),
        solve_eikonal(
            neighbor + IVec2::new(0, 1),
            neighbor + IVec2::new(1, 0),
            resolution,
            distances,
            flags,
        ),
        solve_eikonal(
            neighbor + IVec2::new(0, -1),
            neighbor + IVec2::new(1, 0),
            resolution,
            distances,
            flags,
        ),
        solve_eikonal(
            neighbor + IVec2::new(0, 1),
            neighbor + IVec2::new(-1, 0),
            resolution,
            distances,
            flags,
        ),
    ];
    Some(Vec4::from_slice(&eikonals).min_element())
}

fn inpaint_pixel(
    image: &Image<f32>,
    coordinate: USizeVec2,
    resolution: USizeVec2,
    distances: &mut DistanceArray,
    flags: &mut FlagArray,
    radius: i32,
) -> Array1<f32> {
    let distance = distances[[coordinate.y, coordinate.x]];
    let gradient_distance = pixel_gradient(coordinate, resolution, distances, flags);

    let mut weight_sum = 0.0;
    let channels = image.dim().2;
    let mut output_pixel = arr1(&vec![0.0; channels]);
    for y in -radius..=radius {
        for x in -radius..=radius {
            let current_coordinate = coordinate.as_ivec2() + IVec2::new(x, y);
            if current_coordinate.y < 0
                || current_coordinate.y >= resolution.y as i32
                || current_coordinate.x < 0
                || current_coordinate.x >= resolution.x as i32
            {
                continue;
            }
            let neighbor = current_coordinate.as_usizevec2();
            if flags[[neighbor.y, neighbor.x]] == Flag::Inside {
                continue;
            }
            let direction = coordinate.as_ivec2() - neighbor.as_ivec2();
            let length_pow = powi(direction.x as f32, 2) + powi(direction.y as f32, 2);
            let length = sqrt(length_pow);
            if length > radius as f32 {
                continue;
            }

            let mut direction_factor = (direction.y as f32 * gradient_distance.y
                + direction.x as f32 * gradient_distance.x)
                .abs();
            if direction_factor == 0.0 {
                direction_factor = f32::EPSILON;
            }

            let neighbor_distance = distances[[neighbor.y, neighbor.x]];
            let level_factor = 1.0 / (1.0 + (neighbor_distance - distance).abs());
            let distance_factor = 1.0 / (length * length_pow);
            let weight = (direction_factor * distance_factor * level_factor).abs();
            for (channel, value) in output_pixel.iter_mut().enumerate() {
                *value += weight
                    * image[[
                        current_coordinate.y as usize,
                        current_coordinate.x as usize,
                        channel,
                    ]];
            }
            weight_sum += weight;
        }
    }
    for channel in output_pixel.iter_mut() {
        *channel /= weight_sum;
    }
    output_pixel
}

/// Data structure that stores the processing data.
struct ProcessData {
    distances: DistanceArray,
    process_image: Image<f32>,
    flags: FlagArray,
    heap: BinaryHeap<Reverse<QueueItem>>,
}

impl ProcessData {
    /// Initialize the process data and precompute the distances, flags and fill heap
    pub fn new<ImageType, MaskType>(
        resolution: USizeVec2,
        image: &ArrayViewMut3<ImageType>,
        mask: &ArrayView2<MaskType>,
        radius: i32,
    ) -> Result<Self>
    where
        ImageType: AsPrimitive<f32> + Copy,
        MaskType: AsPrimitive<f32> + Copy + 'static,
    {
        let mut distances = Array2::<f32>::from_elem((resolution.y, resolution.x), MAX);
        let process_image: Image<f32> = image.mapv(|pixel| pixel.as_());
        let mask_array = convert_mask_to_flag_array(mask, resolution);
        let mut flags = mask_array
            .to_owned()
            .mapv(|f| if f == Flag::Band { Flag::Inside } else { f });
        let mut heap = BinaryHeap::new();
        let non_zero: Vec<_> = flags
            .indexed_iter()
            .filter_map(|(index, &item)| {
                if item != Flag::Known {
                    Some(index)
                } else {
                    None
                }
            })
            .collect();

        for index in non_zero.iter() {
            let coordinates = USizeVec2::new(index.1, index.0);
            let neighbors = get_neighbors(coordinates.as_ivec2());
            for neighbor in neighbors {
                if neighbor.y < 0
                    || neighbor.y >= resolution.y as i32
                    || neighbor.x < 0
                    || neighbor.x >= resolution.x as i32
                {
                    continue;
                };
                if flags[[neighbor.y as usize, neighbor.x as usize]] == Flag::Band {
                    continue;
                }

                if mask_array[[neighbor.y as usize, neighbor.x as usize]] == Flag::Known {
                    flags[[neighbor.y as usize, neighbor.x as usize]] = Flag::Band;
                    distances[[neighbor.y as usize, neighbor.x as usize]] = 0.0;
                    heap.push(Reverse(QueueItem::new(0.0, neighbor.as_usizevec2())));
                }
            }
        }

        compute_outside_distances(resolution, &mut distances, &flags, &heap, radius)?;

        Ok(Self {
            distances,
            process_image,
            flags,
            heap,
        })
    }
}

/// ## Inpaint the input image according to the mask provided.
///
/// 3d arrays are expected for inpainting of the image, while 2d array is expected for mask.
/// As the mask consists of only one mask channel.
///
/// In the image array, the rows is the height, the columns is the width
/// and the dimensions are the channels.
///
/// ### Arguments:
///
/// * `image`: array to inpaint.
/// * `mask`: mask that defines the region that will be inpainted
/// * `radius`: radius of near pixels that are considered for npainting.
///
/// ### Example
/// ```rust
/// use inpaint::telea_inpaint;
/// use ndarray::{Array2, Array3};
/// use glam::USizeVec2;
///
/// let resolution = USizeVec2::new(1920, 1080);
/// // obviously you need to use actual data, this is just an example
/// let mut input_image = Array3::from_elem((resolution.y, resolution.x, 4), 0.0);
/// let mask = Array2::from_elem((resolution.y, resolution.x), 0.0);
///
/// telea_inpaint(&mut input_image.view_mut(), &mask.view(), 1).unwrap();
/// ```
pub fn telea_inpaint<ImageType, MaskType>(
    image: &mut ArrayViewMut3<ImageType>,
    mask: &ArrayView2<MaskType>,
    radius: i32,
) -> Result<()>
where
    ImageType: AsPrimitive<f32> + Copy,
    f32: num_traits::AsPrimitive<ImageType>,
    MaskType: AsPrimitive<f32> + Copy + 'static,
{
    if image.shape()[1] != mask.ncols() || image.shape()[0] != mask.nrows() {
        return Err(Error::DimensionMismatch);
    }

    let resolution = USizeVec2::new(image.shape()[1], image.shape()[0]);
    let mut process_data = ProcessData::new(resolution, image, mask, radius)?;
    while !process_data.heap.is_empty() {
        let coordinates = if let Some(node) = process_data.heap.pop() {
            node.0.coordinates
        } else {
            return Err(Error::HeapDoesNotContainData);
        };
        process_data.flags[[coordinates.y, coordinates.x]] = Flag::Known;

        let neighbors = get_neighbors(coordinates.as_ivec2());

        for neighbor in neighbors {
            if neighbor.y >= resolution.y as i32 || neighbor.x >= resolution.x as i32 {
                continue;
            }

            let distance = match get_eikonal(
                resolution,
                &mut process_data.distances,
                &mut process_data.flags,
                neighbor,
            ) {
                Some(value) => value,
                None => continue,
            };

            process_data.distances[[neighbor.y as usize, neighbor.x as usize]] = distance;
            let pixel = inpaint_pixel(
                &process_data.process_image,
                neighbor.as_usizevec2(),
                resolution,
                &mut process_data.distances,
                &mut process_data.flags,
                radius,
            );
            process_data
                .process_image
                .slice_mut(s![neighbor.y, neighbor.x, ..])
                .assign(&pixel);

            process_data.flags[[neighbor.y as usize, neighbor.x as usize]] = Flag::Band;
            process_data
                .heap
                .push(Reverse(QueueItem::new(distance, neighbor.as_usizevec2())));
        }
    }
    image
        .indexed_iter_mut()
        .for_each(|((y, x, channel), value)| {
            *value = process_data.process_image[[y, x, channel]].as_();
        });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, Pixel, Rgba32FImage};
    use image_ndarray::prelude::*;
    use ndarray::s;
    use rstest::rstest;
    use time::OffsetDateTime;

    #[rstest]
    #[case(
        include_bytes!("../test/images/input/bird.png"),
        include_bytes!("../test/images/mask/thin.png"),
        include_bytes!("../test/images/expected/telea/bird_thin.png")
    )]
    #[case(
        include_bytes!("../test/images/input/bird.png"),
        include_bytes!("../test/images/mask/medium.png"),
        include_bytes!("../test/images/expected/telea/bird_medium.png")
    )]
    #[case(
        include_bytes!("../test/images/input/bird.png"),
        include_bytes!("../test/images/mask/large.png"),
        include_bytes!("../test/images/expected/telea/bird_large.png")
    )]
    #[case(
        include_bytes!("../test/images/input/bird.png"),
        include_bytes!("../test/images/mask/text.png"),
        include_bytes!("../test/images/expected/telea/bird_text.png")
    )]
    #[case(
        include_bytes!("../test/images/input/toad.png"),
        include_bytes!("../test/images/mask/thin.png"),
        include_bytes!("../test/images/expected/telea/toad_thin.png")
    )]
    #[case(
        include_bytes!("../test/images/input/toad.png"),
        include_bytes!("../test/images/mask/medium.png"),
        include_bytes!("../test/images/expected/telea/toad_medium.png")
    )]
    #[case(
        include_bytes!("../test/images/input/toad.png"),
        include_bytes!("../test/images/mask/large.png"),
        include_bytes!("../test/images/expected/telea/toad_large.png")
    )]
    #[case(
        include_bytes!("../test/images/input/toad.png"),
        include_bytes!("../test/images/mask/text.png"),
        include_bytes!("../test/images/expected/telea/toad_text.png")
    )]
    #[case(
        include_bytes!("../test/images/input/pizza.png"),
        include_bytes!("../test/images/mask/rectangle.png"),
        include_bytes!("../test/images/expected/telea/pizza_rectangle.png")
    )]
    #[case(
        include_bytes!("../test/images/input/pizza.png"),
        include_bytes!("../test/images/mask/rectangle-strokes.png"),
        include_bytes!("../test/images/expected/telea/pizza_rectangle-strokes.png")
    )]
    #[case(
        include_bytes!("../test/images/input/pizza.png"),
        include_bytes!("../test/images/mask/rectangle-half.png"),
        include_bytes!("../test/images/expected/telea/pizza_rectangle-half.png")
    )]

    /// Test inpaint of provided image with mask
    fn test_inpaint_f32(#[case] image: &[u8], #[case] mask: &[u8], #[case] expected: &[u8]) {
        let mut image = image::load_from_memory_with_format(image, image::ImageFormat::Png)
            .unwrap()
            .to_rgba32f();
        let mask = image::load_from_memory_with_format(mask, image::ImageFormat::Png)
            .unwrap()
            .to_luma8();

        #[cfg(feature = "std")]
        let start = OffsetDateTime::now_utc();
        telea_inpaint(
            &mut image.as_ndarray_mut(),
            &mask.to_ndarray().slice(ndarray::s![.., .., 0]),
            5,
        )
        .unwrap();

        #[cfg(feature = "std")]
        println!(
            "Duration of inpaint: {} seconds",
            (OffsetDateTime::now_utc() - start).as_seconds_f32()
        );

        let result = DynamicImage::from(image.clone());

        let expected_image = DynamicImage::from(
            image::load_from_memory_with_format(expected, image::ImageFormat::Png).unwrap(),
        )
        .to_rgb8();
        let comparison_score =
            image_compare::rgb_hybrid_compare(&result.to_rgb8(), &expected_image)
                .unwrap()
                .score;

        #[cfg(feature = "std")]
        println!("Test got score: {}", comparison_score);
        assert_eq!(comparison_score, 1.0);
    }

    #[rstest]
    #[case(
        include_bytes!("../test/images/input/bird.png"),
        include_bytes!("../test/images/mask/thin.png"),
        include_bytes!("../test/images/expected/telea/bird_thin.png")
    )]
    #[case(
        include_bytes!("../test/images/input/bird.png"),
        include_bytes!("../test/images/mask/medium.png"),
        include_bytes!("../test/images/expected/telea/bird_medium.png")
    )]
    #[case(
        include_bytes!("../test/images/input/bird.png"),
        include_bytes!("../test/images/mask/large.png"),
        include_bytes!("../test/images/expected/telea/bird_large.png")
    )]
    #[case(
        include_bytes!("../test/images/input/bird.png"),
        include_bytes!("../test/images/mask/text.png"),
        include_bytes!("../test/images/expected/telea/bird_text.png")
    )]
    #[case(
        include_bytes!("../test/images/input/toad.png"),
        include_bytes!("../test/images/mask/thin.png"),
        include_bytes!("../test/images/expected/telea/toad_thin.png")
    )]
    #[case(
        include_bytes!("../test/images/input/toad.png"),
        include_bytes!("../test/images/mask/medium.png"),
        include_bytes!("../test/images/expected/telea/toad_medium.png")
    )]
    #[case(
        include_bytes!("../test/images/input/toad.png"),
        include_bytes!("../test/images/mask/text.png"),
        include_bytes!("../test/images/expected/telea/toad_text.png")
    )]

    /// Test inpaint of provided image with mask
    fn test_inpaint_u8(#[case] image: &[u8], #[case] mask: &[u8], #[case] expected: &[u8]) {
        let mut image = image::load_from_memory_with_format(image, image::ImageFormat::Png)
            .unwrap()
            .to_luma8();
        let mask = image::load_from_memory_with_format(mask, image::ImageFormat::Png)
            .unwrap()
            .to_luma8();

        #[cfg(feature = "std")]
        let start = OffsetDateTime::now_utc();
        telea_inpaint(
            &mut image.as_ndarray_mut(),
            &mask.to_ndarray().slice(ndarray::s![.., .., 0]),
            5,
        )
        .unwrap();

        #[cfg(feature = "std")]
        println!(
            "Duration of inpaint: {} seconds",
            (OffsetDateTime::now_utc() - start).as_seconds_f32()
        );

        let result = DynamicImage::from(image.clone());

        let expected_image = DynamicImage::from(
            image::load_from_memory_with_format(expected, image::ImageFormat::Png).unwrap(),
        )
        .to_rgb8();
        let comparison_score =
            image_compare::rgb_hybrid_compare(&result.to_rgb8(), &expected_image)
                .unwrap()
                .score;

        #[cfg(feature = "std")]
        println!("Test got score: {}", comparison_score);
        assert!(comparison_score >= 0.99); // Slightly lower because of precision
    }

    #[test]
    fn inpaint_rectangular() {
        let resolution = USizeVec2::new(1920, 1080);
        let mut test_shape = Array3::from_elem((resolution.y, resolution.x, 4), 0.0);

        let test_mask = Array2::from_elem((resolution.y, resolution.x), 0.0);

        telea_inpaint(&mut test_shape.view_mut(), &test_mask.view(), 1).unwrap();
    }
}
