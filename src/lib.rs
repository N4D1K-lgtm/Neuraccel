// lib.rs

// External dependencies
extern crate blastoff;
extern crate cudnn;

// Internal modules
mod decoding;
mod encoding;
mod kernels;
mod neurons;
mod simulation;
mod synapses;
mod utils;

// Public interfaces
pub use decoding::*;
pub use encoding::*;
pub use neurons::*;
pub use simulation::*;
pub use synapses::*;
