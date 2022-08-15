use std::ops::{Add, Div, Mul, Rem, Sub};

use ndarray::{aview1, Array1, Array2, ArrayView2, Axis};
use ndarray_stats::{interpolate::Midpoint, Quantile1dExt};
use noisy_float::types::n64;
use num_traits::{FromPrimitive, Zero};

/// The ndarray container for spectral data.
/// Subsequent frequnecy channels are aligned in memory and ndarray is "C/Python" style,
/// as such, this has dimensions (samples, channels)
type Spectra<'a, T> = ArrayView2<'a, T>;

pub trait Filter<T> {
    /// Return the mask of this filter, where `true` indicates this channel should be removed
    fn mask(&self, spectra: Spectra<T>) -> Array2<bool>;
}

/// Computes the "bandpass" of the `spectra`.
/// This is the mean of the dynamic spectra across the time axis to get the average power in each channel.
/// This won't be super fast because the data is frequency-major.
pub fn bandpass<T>(spectra: &Spectra<T>) -> Array1<T>
where
    T: Clone + Zero + FromPrimitive + Add<Output = T> + Div<Output = T>,
{
    spectra.mean_axis(Axis(0)).unwrap()
}

/// Creates a 2D `Spectra` from an array of raw measurements
pub fn to_spectra<T>(raw_spectra: &[T], channels: usize) -> Spectra<T> {
    let samples = raw_spectra.len() / channels;
    aview1(raw_spectra).into_shape((samples, channels)).unwrap()
}

/// A filter to remove system-temperature-based bandpass. `tolerance` indicated what fraction of the median to clip.
pub struct Tsys {
    pub tolerance: f32,
}

impl<T> Filter<T> for Tsys
where
    T: Clone
        + Zero
        + FromPrimitive
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Mul<Output = T>
        + Ord
        + Rem<Output = T>,
    f32: std::convert::From<T>,
{
    fn mask(&self, spectra: Spectra<T>) -> Array2<bool> {
        let mut t_sys = bandpass(&spectra);
        let t_sys_median = f32::from(t_sys.quantile_mut(n64(0.5), &Midpoint).unwrap());
        let t_sys_mask = t_sys.mapv(|v| f32::from(v) < self.tolerance * t_sys_median);
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_to_spectra() {
        let spectra = vec![0u16; 2048 * 16384];
        to_spectra(&spectra, 2048);
    }

    #[test]
    fn test_bandpass() {
        let raw: Vec<u16> = vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
        let spectra = to_spectra(&raw, 4);
        let bp = bandpass(&spectra);
        assert_eq!(bp, array![1, 2, 3, 4]);
    }
}
