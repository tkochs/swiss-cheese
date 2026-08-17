mod errors;

pub use errors::Errors;

pub struct SendPtr(pub *mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
