use ndarray::ArrayView2;
use rayon::prelude::*;

pub struct SendPtr(pub *mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
