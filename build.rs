// build.rs

use glob::glob;

fn main() {
    let glog_url = "https://github.com/google/glog/archive/refs/tags/v0.5.0.tar.gz";
    let glog_hash = "2368e3e0a95cce8b5b35a133271b480f";
    let dst = cmake::Config::new("external/stub")
        .define("GLOG_URL", glog_url)
        .define("GLOG_HASH", glog_hash)
        .generator("Ninja")
        .build();
    let mut oneflow_common = cc::Build::new();
    for entry in glob("oneflow/core/common/**/*.cpp").expect("Failed to read glob pattern") {
        match entry {
            Ok(path) => {
                oneflow_common.file(path);
            }
            Err(e) => println!("{:?}", e),
        }
    }
    oneflow_common.include(".").compile("oneflow_common");
    println!("cargo:rerun-if-changed=src/hello.c");
}
