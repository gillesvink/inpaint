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
use core::f32;
use glam::{IVec2, USizeVec2, Vec2, Vec4};
use ndarray::{Array1, Array2, Array3, arr1, s};
use num_traits::AsPrimitive;
use std::cmp::Reverse;
use std::{cmp::Ordering, collections::BinaryHeap};

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
        let distance = 2.0 - (a_distance - b_distance).powf(2.0);
        if distance > 0.0 {
            let r = distance.sqrt();
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

    Vec2::new(
        calculate_gradient(
            resolution.x,
            distances,
            flags,
            distance,
            coordinates.x,
            coordinates.y,
        ),
        calculate_gradient(
            resolution.y,
            distances,
            flags,
            distance,
            coordinates.y,
            coordinates.x,
        ),
    )
}

/// Calculate gradient weighting
fn calculate_gradient(
    size: usize,
    distances: &DistanceArray,
    flags: &FlagArray,
    value: f32,
    a: usize,
    b: usize,
) -> f32 {
    let next = a + 1;
    if next >= size || a == 0 {
        return MAX;
    }

    let previous = a - 1;

    let gradient;
    let flag_previous = flags[[previous, b]];
    let flag_next = flags[[next, b]];

    if flag_previous != Flag::Inside && flag_next != Flag::Inside {
        gradient = (distances[[next, b]] - distances[[previous, b]]) / 2.0;
    } else if flag_previous != Flag::Inside {
        gradient = value - distances[[previous, b]];
    } else if flag_next != Flag::Inside {
        gradient = distances[[next, b]] - value;
    } else {
        gradient = 0.0;
    }

    gradient
}

/// Normalize value to 0-1 range in float
fn normalize_value<P>(value: P) -> f32
where
    P: AsPrimitive<f32>,
{
    value.as_()
        / match std::any::TypeId::of::<P>() {
            id if id == std::any::TypeId::of::<u8>() => u8::MAX as f32,
            id if id == std::any::TypeId::of::<u16>() => u16::MAX as f32,
            id if id == std::any::TypeId::of::<u32>() => u32::MAX as f32,
            id if id == std::any::TypeId::of::<u32>() => u64::MAX as f32,
            id if id == std::any::TypeId::of::<u32>() => u128::MAX as f32,
            id if id == std::any::TypeId::of::<i8>() => i8::MAX as f32,
            id if id == std::any::TypeId::of::<i16>() => i16::MAX as f32,
            id if id == std::any::TypeId::of::<i32>() => i32::MAX as f32,
            id if id == std::any::TypeId::of::<i32>() => i64::MAX as f32,
            id if id == std::any::TypeId::of::<i32>() => i128::MAX as f32,
            _ => 1.0,
        }
}

/// Convert the input array of any type to the FlagArray (which consists of enum values)
fn convert_mask_to_flag_array<P>(mask: &Array2<P>, resolution: USizeVec2) -> FlagArray
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
        || neighbor.y > resolution.y as i32
        || neighbor.x < 0
        || neighbor.x > resolution.x as i32
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
                || current_coordinate.y > resolution.y as i32
                || current_coordinate.x < 0
                || current_coordinate.x > resolution.x as i32
            {
                continue;
            }
            let neighbor = current_coordinate.as_usizevec2();
            if flags[[neighbor.y, neighbor.x]] == Flag::Inside {
                continue;
            }
            let direction = coordinate.as_ivec2() - neighbor.as_ivec2();
            let length_pow = (direction.x as f32).powi(2) + (direction.y as f32).powi(2);
            let length = length_pow.sqrt();
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
    for i in output_pixel.iter_mut() {
        *i /= weight_sum;
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
        image: &Image<ImageType>,
        mask: &Array2<MaskType>,
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
            .clone()
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

/// Inpaint the input image according to the mask provided.
pub fn telea_inpaint<ImageType, MaskType>(
    image: &mut Image<ImageType>,
    mask: Array2<MaskType>,
    radius: i32,
) -> Result<()>
where
    ImageType: AsPrimitive<f32> + Copy,
    f32: num_traits::AsPrimitive<ImageType>,
    MaskType: AsPrimitive<f32> + Copy + 'static,
{
    if image.shape()[0] != mask.ncols() || image.shape()[1] != mask.nrows() {
        return Err(Error::DimensionMismatch);
    }

    let resolution = USizeVec2::new(image.shape()[1], image.shape()[0]);
    let mut process_data = ProcessData::new(resolution, image, &mask, radius)?;
    while !process_data.heap.is_empty() {
        let coordinates = if let Some(node) = process_data.heap.pop() {
            node.0.coordinates
        } else {
            return Err(Error::HeapDoesNotContainData);
        };
        process_data.flags[[coordinates.y, coordinates.x]] = Flag::Known;

        let neighbors = get_neighbors(coordinates.as_ivec2());

        for neighbor in neighbors {
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
                .slice_mut(s![neighbor.y, neighbor.x, 0..])
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
    use ndarray::s;
    use rstest::rstest;
    use std::path::PathBuf;
    use std::time::Instant;

    /// Just a utility to make it easier to save the test results
    fn store_test_result(image: DynamicImage, path: PathBuf) {
        image.to_rgb8().save(path).unwrap();
    }

    fn load_test_image(path: PathBuf) -> Rgba32FImage {
        let image = image::open(path);
        image.unwrap().to_rgba32f()
    }

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

    /// Test inpaint of provided image with mask
    fn test_inpaint_f32(#[case] image: PathBuf, #[case] mask: PathBuf, #[case] expected: PathBuf) {
        let mut image = image::open(image).unwrap().to_rgba32f();
        let (width, height) = image.dimensions();
        let resolution = USizeVec2::new(width as usize, height as usize);
        let mask = image::open(mask).unwrap().to_luma8();
        let mut input_image: Image<f32> =
            Image::from_shape_fn((resolution.x, resolution.y, 4), |(y, x, channel)| {
                image.get_pixel(x as u32, y as u32).0[channel]
            });
        let input_mask: Array2<u8> =
            Array2::from_shape_fn((resolution.y, resolution.x), |(y, x)| {
                mask.get_pixel(x as u32, y as u32)[0]
            });

        let start = Instant::now();
        telea_inpaint(&mut input_image, input_mask, 5).unwrap();

        println!("Duration of inpaint: {:?}", start.elapsed());

        for (x, y, pixel) in image.enumerate_pixels_mut() {
            let data = input_image.slice(s![y as usize, x as usize, ..]);
            pixel
                .channels_mut()
                .copy_from_slice(data.as_slice().unwrap());
        }
        let result = DynamicImage::from(image.clone());

        if !expected.exists() {
            store_test_result(result.clone(), expected.clone());
        }

        let expected_image = DynamicImage::from(load_test_image(expected)).to_rgb8();
        let comparison_score =
            image_compare::rgb_hybrid_compare(&result.to_rgb8(), &expected_image)
                .unwrap()
                .score;

        println!("Test got score: {}", comparison_score);
        assert_eq!(comparison_score, 1.0);
    }

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

    /// Test inpaint of provided image with mask
    fn test_inpaint_u8(#[case] image: PathBuf, #[case] mask: PathBuf, #[case] expected: PathBuf) {
        let mut image = image::open(image).unwrap().to_rgba8();
        let (width, height) = image.dimensions();
        let resolution = USizeVec2::new(width as usize, height as usize);
        let mask = image::open(mask).unwrap().to_luma8();
        let input_mask: Array2<u8> =
            Array2::from_shape_fn((resolution.x, resolution.y), |(y, x)| {
                mask.get_pixel(x as u32, y as u32)[0]
            });

        let mut input_image: Image<u8> =
            Image::from_shape_fn((resolution.x, resolution.y, 4), |(y, x, channel)| {
                image.get_pixel(x as u32, y as u32).0[channel]
            });

        let start = Instant::now();
        telea_inpaint(&mut input_image, input_mask, 5).unwrap();
        println!("Duration of inpaint: {:?}", start.elapsed());

        image.copy_from_slice(input_image.as_slice().unwrap());
        let result = DynamicImage::from(image.clone());

        if !expected.exists() {
            store_test_result(result.clone(), expected.clone());
        }

        let expected_image = DynamicImage::from(load_test_image(expected)).to_rgb8();
        let comparison_score =
            image_compare::rgb_hybrid_compare(&result.to_rgb8(), &expected_image)
                .unwrap()
                .score;

        println!("Test got score: {}", comparison_score);
        assert!(comparison_score >= 0.99); // Slightly lower because of precision
    }
}
