extern crate bindgen;
extern crate clang_sys;

use std::{env, path::PathBuf, process::Command};

fn main() {
    init();
    if cfg!(feature = "cuda") {
        build_cuda();
    }

    if cfg!(feature = "opencl") {
        build_opencl();
    }

    // Add other backends as needed...
}
fn init() {
    clang_sys::load().expect("Unable to find libclang");
}
fn build_cuda() {
    // Path to the C header file
    let header_path = "external/cuda/cuda_kernels.h";

    // Path to the CUDA source file
    let cuda_src = "external/cuda/cuda_kernels.cu";

    // Path to the output shared library
    let shared_lib = "external/cuda/libcuda.so";

    // Generate the bindings
    let bindings = bindgen::Builder::default()
        .header(header_path)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Failed to generate bindings");

    // Write the bindings to a file
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Failed to write bindings");

    // Compile the CUDA code into a shared library

    let mut command = Command::new("nvcc");
    command
        .arg(cuda_src)
        .arg("-shared")
        .arg("-o")
        .arg(shared_lib);

    if cfg!(unix) {
        command.arg("-fPIC");
    }

    let status = command.status().expect("Failed to execute nvcc");

    if !status.success() {
        panic!("CUDA compilation failed");
    }

    // Print the link flags for Cargo
    println!("cargo:rustc-link-search=native=external/cuda");
    println!("cargo:rustc-link-lib=dylib=cuda");
}

fn build_opencl() {
    // Build logic for OpenCL goes here...
    // You can specify a different folder in the "external" directory
}
