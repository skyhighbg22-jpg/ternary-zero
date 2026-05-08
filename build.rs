use std::env;
use std::path::PathBuf;
use std::process::Command;

fn find_nvcc() -> PathBuf {
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let nvcc_name = if cfg!(target_os = "windows") { "nvcc.exe" } else { "nvcc" };
        let nvcc = PathBuf::from(&cuda_home).join("bin").join(nvcc_name);
        if nvcc.exists() {
            return nvcc;
        }
    }
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let nvcc_name = if cfg!(target_os = "windows") { "nvcc.exe" } else { "nvcc" };
        let nvcc = PathBuf::from(&cuda_path).join("bin").join(nvcc_name);
        if nvcc.exists() {
            return nvcc;
        }
    }
    PathBuf::from(if cfg!(target_os = "windows") { "nvcc.exe" } else { "nvcc" })
}

fn find_cuda_lib_path() -> Option<PathBuf> {
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let lib_path = PathBuf::from(&cuda_home).join("lib").join("x64");
        if lib_path.exists() {
            return Some(lib_path);
        }
    }
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let lib_path = PathBuf::from(&cuda_path).join("lib").join("x64");
        if lib_path.exists() {
            return Some(lib_path);
        }
    }
    None
}

fn find_msvc_bin() -> Option<PathBuf> {
    // Try using the cc crate to find the MSVC compiler
    let build = cc::Build::new();
    let compiler = build.get_compiler();
    compiler.path().parent().map(|p| p.to_path_buf())
}

fn main() {
    let kernel_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("kernel");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=kernel/ternary_zero.cu");
    println!("cargo:rerun-if-changed=kernel/ternary_zero.h");
    println!("cargo:rerun-if-changed=kernel/ptx_utils.h");
    println!("cargo:rerun-if-changed=build.rs");

    let nvcc = find_nvcc();

    // Compile .cu to .obj (Windows) or .o (Linux) using nvcc directly
    let obj_ext = if cfg!(target_os = "windows") { "obj" } else { "o" };
    let obj_path = out_dir.join(format!("ternary_zero.{}", obj_ext));

    let mut cmd = Command::new(&nvcc);
    cmd.arg("-O3")
        .arg("--use_fast_math")
        .arg("-lineinfo")
        .arg("-maxrregcount=64")
        .arg("--gpu-architecture=sm_89")
        .arg("-std=c++17")
        .arg("-c")
        .arg(format!("-I{}", kernel_dir.display()))
        .arg("-o")
        .arg(&obj_path)
        .arg(kernel_dir.join("ternary_zero.cu"));

    // On Windows, nvcc needs cl.exe in PATH. Use cc crate to find MSVC tools dir.
    if cfg!(target_os = "windows") {
        if let Some(msvc_bin) = find_msvc_bin() {
            let current_path = env::var("PATH").unwrap_or_default();
            let new_path = format!("{};{}", msvc_bin.display(), current_path);
            cmd.env("PATH", &new_path);
        }
    }

    println!("cargo:warning=Running: {:?}", cmd);

    let status = cmd.status().expect("Failed to execute nvcc");
    if !status.success() {
        panic!("nvcc compilation failed with status: {}", status);
    }

    // Create a static library from the compiled object
    if cfg!(target_os = "windows") {
        // Use MSVC lib.exe - find it in the same dir as cl.exe
        let lib_path = out_dir.join("ternary_zero.lib");
        let mut lib_cmd = if let Some(msvc_bin) = find_msvc_bin() {
            let lib_exe = msvc_bin.join("lib.exe");
            let mut c = Command::new(&lib_exe);
            let current_path = env::var("PATH").unwrap_or_default();
            c.env("PATH", format!("{};{}", msvc_bin.display(), current_path));
            c
        } else {
            Command::new("lib")
        };
        lib_cmd
            .arg(format!("/OUT:{}", lib_path.display()))
            .arg(&obj_path);

        let status = lib_cmd.status().expect("Failed to execute lib.exe");
        if !status.success() {
            panic!("lib.exe failed to create static library");
        }
    } else {
        let lib_path = out_dir.join("libternary_zero.a");
        let status = Command::new("ar")
            .arg("rcs")
            .arg(&lib_path)
            .arg(&obj_path)
            .status()
            .expect("Failed to execute ar");
        if !status.success() {
            panic!("ar failed to create static library");
        }
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=ternary_zero");

    if let Some(cuda_lib) = find_cuda_lib_path() {
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    }

    println!("cargo:rustc-link-lib=static=cudart_static");

    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=dylib=msvcprt");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}
