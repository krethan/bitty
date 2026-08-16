use bitllm_runtime::gguf::{to_torch_layout, GgufLoader};

fn dump(t: &bitllm_tensor::Tensor, label: &str, stride: usize) {
    let s = t.as_f32_slice();
    for r in 0..2 {
        let row = &s[r * stride..(r + 1) * stride];
        let maxa = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        println!(
            "{} row{} max_abs={:.6} first8={:?}",
            label,
            r,
            maxa,
            &row[..8]
        );
    }
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dump_gguf <file.gguf>");
    let loader = GgufLoader::load(&path)?;
    for name in loader.tensor_names() {
        let t = loader.load_tensor(name)?;
        let t2 = to_torch_layout(t.clone());
        match name {
            "token_embd.weight" => {
                let s = t.as_f32_slice();
                println!("raw token_embd shape={:?} data[0]={} data[1]={} data[576]={} data[49152]={} data[98304]={}",
                    t.shape(), s[0], s[1], s[576], s[49152], s[98304]);
                dump(&t2, "token_embd", 576);
                println!("  raw shape={:?} torch shape={:?}", t.shape(), t2.shape());
            }
            "output_norm.weight" => {
                let s = t.as_f32_slice();
                println!("output_norm first8={:?}", &s[..8]);
            }
            "blk.0.attn_q.weight" => {
                let s = t2.as_f32_slice();
                println!(
                    "blk0.attn_q torch shape={:?} row0 first8={:?}",
                    t2.shape(),
                    &s[..8]
                );
            }
            "blk.0.attn_k.weight" => {
                let s = t2.as_f32_slice();
                println!(
                    "blk0.attn_k torch shape={:?} row0 first8={:?}",
                    t2.shape(),
                    &s[..8]
                );
            }
            "blk.0.attn_v.weight" => {
                let s = t2.as_f32_slice();
                println!(
                    "blk0.attn_v torch shape={:?} row0 first8={:?}",
                    t2.shape(),
                    &s[..8]
                );
            }
            "blk.0.attn_output.weight" => {
                let s = t2.as_f32_slice();
                println!(
                    "blk0.attn_output torch shape={:?} row0 first8={:?}",
                    t2.shape(),
                    &s[..8]
                );
            }
            "blk.0.ffn_gate.weight" => {
                let s = t2.as_f32_slice();
                println!(
                    "blk0.ffn_gate torch shape={:?} row0 first8={:?}",
                    t2.shape(),
                    &s[..8]
                );
            }
            "blk.0.ffn_up.weight" => {
                let s = t2.as_f32_slice();
                println!(
                    "blk0.ffn_up torch shape={:?} row0 first8={:?}",
                    t2.shape(),
                    &s[..8]
                );
            }
            "blk.0.ffn_down.weight" => {
                let s = t2.as_f32_slice();
                println!(
                    "blk0.ffn_down torch shape={:?} row0 first8={:?}",
                    t2.shape(),
                    &s[..8]
                );
            }
            "blk.0.attn_norm.weight" => {
                let s = t.as_f32_slice();
                println!("blk0.attn_norm first8={:?}", &s[..8]);
            }
            "blk.0.ffn_norm.weight" => {
                let s = t.as_f32_slice();
                println!("blk0.ffn_norm first8={:?}", &s[..8]);
            }
            _ => {}
        }
    }
    Ok(())
}
