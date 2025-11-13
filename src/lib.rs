#![doc = include_str!("../README.md")]
#![allow(unused_imports)]
#![cfg_attr(not(feature = "std"), no_std)]

mod error;
pub use error::Error;
pub mod prelude;
mod telea;
pub use prelude::*;
pub use telea::telea_inpaint;

#[cfg(feature = "python-bindings")]
#[pyo3::pymodule]
mod inpaint {
    use crate::error::Result;
    use numpy::IntoPyArray;
    use numpy::{PyArray3, PyReadonlyArray2, PyReadonlyArray3};
    use pyo3::Python;
    use pyo3::prelude::*;

    fn telea_inpaint_inner_py<'py, T>(
        py: Python<'py>,
        image: PyReadonlyArray3<'py, T>,
        mask: PyReadonlyArray2<'py, T>,
        radius: i32,
    ) -> Result<Bound<'py, PyArray3<T>>>
    where
        T: numpy::Element + Clone + Copy + num_traits::AsPrimitive<f32> + 'static,
        f32: num_traits::AsPrimitive<T> + Clone + Copy,
    {
        let mut original_image = image.as_array().to_owned();
        let mask_image = mask.as_array().to_owned();

        crate::telea::telea_inpaint(&mut original_image.view_mut(), &mask_image.view(), radius)?;

        Ok(original_image.into_pyarray(py))
    }

    #[pyfunction]
    #[pyo3(name = "telea_inpaint")]
    fn telea_inpaint_py<'py>(
        py: Python<'py>,
        image: PyReadonlyArray3<'py, f32>,
        mask: PyReadonlyArray2<'py, f32>,
        radius: i32,
    ) -> Result<Bound<'py, PyArray3<f32>>> {
        telea_inpaint_inner_py::<f32>(py, image, mask, radius)
    }
}
