// build.rs

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

use glob::glob;
use std::collections::HashSet;

fn main() {
    let cfg_proto_paths = HashSet::from([
        "oneflow/core/common/error.proto",
        "oneflow/core/vm/instruction.proto",
        "oneflow/core/job/job_conf.proto",
        "oneflow/core/job/placement.proto",
        "oneflow/core/operator/op_conf.proto",
        "oneflow/core/operator/interface_blob_conf.proto",
        "oneflow/core/common/shape.proto",
        "oneflow/core/record/record.proto",
        "oneflow/core/job/resource.proto",
        "oneflow/core/register/logical_blob_id.proto",
        "oneflow/core/register/tensor_slice_view.proto",
        "oneflow/core/common/range.proto",
        "oneflow/core/framework/user_op_conf.proto",
        "oneflow/core/framework/user_op_attr.proto",
        "oneflow/core/job/sbp_parallel.proto",
        "oneflow/core/graph/boxing/collective_boxing.proto",
        "oneflow/core/register/blob_desc.proto",
        "oneflow/core/job/scope.proto",
        "oneflow/core/job/mirrored_parallel.proto",
        "oneflow/core/operator/op_attribute.proto",
        "oneflow/core/operator/arg_modifier_signature.proto",
        "oneflow/core/job/blob_lifetime_signature.proto",
        "oneflow/core/job/parallel_signature.proto",
        "oneflow/core/job/parallel_conf_signature.proto",
        "oneflow/core/job/cluster_instruction.proto",
        "oneflow/core/job/initializer_conf.proto",
        "oneflow/core/job/regularizer_conf.proto",
        "oneflow/core/job/learning_rate_schedule_conf.proto",
        "oneflow/core/common/cfg_reflection_test.proto",
        "oneflow/core/common/data_type.proto",
        "oneflow/core/common/device_type.proto",
        "oneflow/core/serving/saved_model.proto",
    ]);
    let glog_url = "https://github.com/google/glog/archive/refs/tags/v0.5.0.tar.gz";
    let glog_hash = "2368e3e0a95cce8b5b35a133271b480f";
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_include: PathBuf = [out_dir.to_str().unwrap(), "include"].iter().collect();
    cmake::Config::new("external/stub")
        .define("GLOG_URL", glog_url)
        .define("GLOG_HASH", glog_hash)
        .generator("Ninja")
        .build();

    // generate c++ and python from proto
    let protoc_path = Path::new(&out_dir).join("bin").join("protoc");
    let proto_include_path = Path::new(&out_dir).join("include");
    for entry in glob("oneflow/core/**/*.proto").expect("Failed to read glob pattern") {
        match entry {
            Ok(path) => {
                println!("cargo:rerun-if-changed={}", path.to_str().unwrap());
                // TODO: create __init__.py for pb
                let status = Command::new(protoc_path.to_str().unwrap())
                    .args(&[
                        "-I",
                        proto_include_path.to_str().unwrap(),
                        "-I",
                        "./",
                        "--cpp_out",
                        out_dir.to_str().unwrap(),
                        "--python_out",
                        "python",
                        "--python_out",
                        out_dir.to_str().unwrap(), // this is for cfg to use in reflection
                        path.to_str().unwrap(),
                    ])
                    .status()
                    .expect("failed to execute process");
                assert!(status.success());
            }
            Err(e) => println!("{:?}", e),
        }
    }
    // generate cfg, because cfg doesn't have include mechanism so must compile all protobuf before
    for entry in glob("oneflow/core/**/*.proto").expect("Failed to read glob pattern") {
        match entry {
            Ok(path) => {
                println!("cargo:rerun-if-changed={}", path.to_str().unwrap());
                if cfg_proto_paths.contains(path.to_str().unwrap()) {
                    let status = Command::new("python3")
                        .args(&[
                            "tools/cfg/template_convert.py",
                            "--project_build_dir",
                            out_dir.to_str().unwrap(),
                            "--of_cfg_proto_python_dir",
                            out_dir.to_str().unwrap(),
                            "--generate_file_type=cfg.cpp",
                            "--proto_file_path",
                            path.to_str().unwrap(),
                        ])
                        .status()
                        .expect("failed to execute process");
                    assert!(status.success());
                }
            }
            Err(e) => println!("{:?}", e),
        }
    }

    // build oneflow common
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
        .include(out_dir)
        .include("./tools/cfg/include")
        .compile("oneflow_common");
}
