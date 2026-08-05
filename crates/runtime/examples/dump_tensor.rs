//! Dump a single GGUF tensor's f32 values to a binary file for cross-checking.
use bitllm_runtime::gguf::{to_torch_layout, GgufLoader};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let gguf_path = PathBuf::from(args.next().expect("usage: dump_tensor <model.gguf> <tensor_name> <out.bin>"));
    let tensor_name = args.next().expect("tensor name");
    let out = args.next().expect("output path");

    let mut reader = GgufLoader::load(&gguf_path)?;
    let t = reader.load_tensor(&tensor_name)?;
    let t = to_torch_layout(t);
    let s = t.as_f32_slice();
    std::fs::write(&out, unsafe {
        std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4)
    })?;
    println!("wrote {} f32 values to {}", s.len(), out);
    Ok(())
}
