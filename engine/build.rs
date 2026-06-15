use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let cu_file = "cuda/kernel.cu";
    let obj_file = out_dir.join("kernel.o");
    let lib_file = out_dir.join("libkernel.a");

    // 1. Compile CUDA -> object file
    let status = Command::new("nvcc")
        .args([
            "-c",
            cu_file,
            "-o",
        ])
        .arg(&obj_file)
        .status()
        .expect("Failed to run nvcc");

    assert!(status.success(), "nvcc failed");

    // 2. Archive into static library
    let status = Command::new("ar")
        .args([
            "crus",
            lib_file.to_str().unwrap(),
            obj_file.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run ar");

    assert!(status.success(), "ar failed");

    // 3. Link search path
    println!("cargo:rustc-link-search=native={}", out_dir.display());

    // 4. Link correct library name (libkernel.a)
    println!("cargo:rustc-link-lib=static=kernel");

    // 5. CUDA runtime
    println!("cargo:rustc-link-lib=dylib=cudart");

    // 6. CUDA library path
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");

    // 7. Rebuild trigger
    println!("cargo:rerun-if-changed={}", cu_file);
}