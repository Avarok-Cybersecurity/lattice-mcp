//! Float transcendentals. `core` has no `exp`/`sqrt`/`powf`; with `std` these
//! call std (bit-for-bit what the code computed before extraction), without
//! it they call `libm`.

#[cfg(feature = "std")]
pub(crate) fn exp(x: f64) -> f64 {
    x.exp()
}
#[cfg(not(feature = "std"))]
pub(crate) fn exp(x: f64) -> f64 {
    libm::exp(x)
}

#[cfg(feature = "std")]
pub(crate) fn powf(base: f64, e: f64) -> f64 {
    base.powf(e)
}
#[cfg(not(feature = "std"))]
pub(crate) fn powf(base: f64, e: f64) -> f64 {
    libm::pow(base, e)
}

#[cfg(feature = "std")]
pub(crate) fn sqrtf(x: f32) -> f32 {
    x.sqrt()
}
#[cfg(not(feature = "std"))]
pub(crate) fn sqrtf(x: f32) -> f32 {
    libm::sqrtf(x)
}
