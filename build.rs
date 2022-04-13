// build.rs

use std::{env, path::PathBuf};

use glob::glob;

fn main() {
    let glog_url = "https://github.com/google/glog/archive/refs/tags/v0.5.0.tar.gz";
    let glog_hash = "2368e3e0a95cce8b5b35a133271b480f";
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_include: PathBuf = [out_dir.to_str().unwrap(), "include"].iter().collect();
    let dst = cmake::Config::new("external/stub")
        .define("GLOG_URL", glog_url)
        .define("GLOG_HASH", glog_hash)
        .generator("Ninja")
        .build();
    let mut oneflow_common = cc::Build::new();
    for entry in glob("oneflow/core/common/**/*.cpp").expect("Failed to read glob pattern") {
        match entry {
            Ok(path) => {
                if !path.to_str().unwrap().ends_with("test.cpp") {
                    oneflow_common.file(path);
                }
            }
            Err(e) => println!("{:?}", e),
        }
    }
    oneflow_common
        .include(".")
        .include(out_include)
        .compile("oneflow_common");
    println!("cargo:rerun-if-changed=src/hello.c");
}
