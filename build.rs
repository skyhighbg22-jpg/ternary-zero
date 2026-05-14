use std::env;
use std::path::PathBuf;
use std::process::Command;

fn find_nvcc() -> PathBuf {
    if let Ok(cuda_home) = env::var("CUDA_HOME") {
        let nvcc_name = if cfg!(target_os = "windows") {
            "nvcc.exe"
        } else {
            "nvcc"
        };
        let nvcc = PathBuf::from(&cuda_home).join("bin").join(nvcc_name);
        if nvcc.exists() {
            return nvcc;
        }
    }
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let nvcc_name = if cfg!(target_os = "windows") {
            "nvcc.exe"
        } else {
            "nvcc"
        };
        let nvcc = PathBuf::from(&cuda_path).join("bin").join(nvcc_name);
        if nvcc.exists() {
            return nvcc;
        }
    }
    PathBuf::from(if cfg!(target_os = "windows") {
        "nvcc.exe"
    } else {
        "nvcc"
    })
}

fn cuda_is_available() -> bool {
    let nvcc = find_nvcc();
    Command::new(&nvcc)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
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
    let build = cc::Build::new();
    let compiler = build.get_compiler();
    compiler.path().parent().map(|p| p.to_path_buf())
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(no_cuda)");

    if cfg!(feature = "cpu-only") {
        println!("cargo:warning=cpu-only feature enabled, skipping CUDA kernel compilation");
        println!("cargo:rustc-cfg=no_cuda");
        return;
    }

    if !cuda_is_available() {
        println!(
            "cargo:warning=CUDA toolkit not found (nvcc not available), skipping CUDA kernel compilation. \
             Use --features cpu-only to silence this warning."
        );
        println!("cargo:rustc-cfg=no_cuda");
        return;
    }

    let kernel_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("kernel");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=kernel/ternary_zero.cu");
    println!("cargo:rerun-if-changed=kernel/ternary_zero.h");
    println!("cargo:rerun-if-changed=kernel/ptx_utils.h");
    println!("cargo:rerun-if-changed=kernel/l2_persist.cu");

    let nvcc = find_nvcc();

    let obj_ext = if cfg!(target_os = "windows") {
        "obj"
    } else {
        "o"
    };

    let mut obj_paths = Vec::new();

    // Compile ternary_zero.cu
    let obj_main = out_dir.join(format!("ternary_zero.{}", obj_ext));
    {
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
            .arg(&obj_main)
            .arg(kernel_dir.join("ternary_zero.cu"));

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
            panic!(
                "nvcc compilation of ternary_zero.cu failed with status: {}",
                status
            );
        }
    }
    obj_paths.push(obj_main);

    // Compile l2_persist.cu
    let obj_l2 = out_dir.join(format!("l2_persist.{}", obj_ext));
    {
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
            .arg(&obj_l2)
            .arg(kernel_dir.join("l2_persist.cu"));

        if cfg!(target_os = "windows") {
            if let Some(msvc_bin) = find_msvc_bin() {
                let current_path = env::var("PATH").unwrap_or_default();
                let new_path = format!("{};{}", msvc_bin.display(), current_path);
                cmd.env("PATH", &new_path);
            }
        }

        println!("cargo:warning=Running: {:?}", cmd);
        let status = cmd
            .status()
            .expect("Failed to execute nvcc for l2_persist.cu");
        if !status.success() {
            panic!(
                "nvcc compilation of l2_persist.cu failed with status: {}",
                status
            );
        }
    }
    obj_paths.push(obj_l2);

    if cfg!(target_os = "windows") {
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
        lib_cmd.arg(format!("/OUT:{}", lib_path.display()));
        for obj in &obj_paths {
            lib_cmd.arg(obj);
        }

        let status = lib_cmd.status().expect("Failed to execute lib.exe");
        if !status.success() {
            panic!("lib.exe failed to create static library");
        }
    } else {
        let lib_path = out_dir.join("libternary_zero.a");
        let mut ar_cmd = Command::new("ar");
        ar_cmd.arg("rcs").arg(&lib_path);
        for obj in &obj_paths {
            ar_cmd.arg(obj);
        }
        let status = ar_cmd.status().expect("Failed to execute ar");
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
