use std::env;
use std::path::PathBuf;

fn find_nvcc() -> String {
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let nvcc = PathBuf::from(&cuda_home).join("bin").join("nvcc");
        if nvcc.exists() {
            return nvcc.to_str().unwrap().to_string();
        }
    }
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let nvcc = PathBuf::from(&cuda_path).join("bin").join("nvcc.exe");
        if nvcc.exists() {
            return nvcc.to_str().unwrap().to_string();
        }
    }
    "nvcc".to_string()
}

fn main() {
    let kernel_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("kernel");

    println!("cargo:rerun-if-changed=kernel/ternary_zero.cu");
    println!("cargo:rerun-if-changed=kernel/ternary_zero.h");
    println!("cargo:rerun-if-changed=kernel/ptx_utils.h");
    println!("cargo:rerun-if-changed=build.rs");

    let nvcc = find_nvcc();

    cc::Build::new()
        .cuda(true)
        .file(kernel_dir.join("ternary_zero.cu"))
        .include(&kernel_dir)
        .flag("-O3")
        .flag("--use_fast_math")
        .flag("-lineinfo")
        .flag("-maxrregcount=64")
        .flag("--gpu-architecture=sm_89")
        .flag("-std=c++17")
        .compile("ternary_zero");

    // Link CUDA runtime statically
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let lib_path = PathBuf::from(&cuda_home).join("lib").join("x64");
        println!("cargo:rustc-link-search=native={}", lib_path.display());
    } else if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let lib_path = PathBuf::from(&cuda_path).join("lib").join("x64");
        println!("cargo:rustc-link-search=native={}", lib_path.display());
    }

    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
