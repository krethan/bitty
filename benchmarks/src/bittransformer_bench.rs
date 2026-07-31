use std::time::Instant;
use bitllm_tensor::{Tensor, DType};
use bitllm_runtime::config::ModelConfig;
use bitllm_runtime::attention::KvCache;
use bitllm_runtime::model::TransformerLayer;
use bitllm_runtime::bittransformer::BitTransformerLayer;

fn main() {
    let config = ModelConfig::tiny_test();
    // Create a tiny FP32 transformer layer and convert to 1-bit
    let fp32_layer = TransformerLayer::new_dummy(&config);
    let bit_layer = BitTransformerLayer::from_fp32_layer(&fp32_layer);

    let seq_len = config.max_seq_len;
    let batch = 16;
    let hidden = config.hidden_size;
    let input = Tensor::zeros(&[batch, seq_len, hidden], DType::F32);

    let mut cache = KvCache::new(config.num_layers, seq_len, config.num_kv_heads(), config.head_dim());

    let start = Instant::now();
    for _ in 0..10 {
        let _ = bit_layer.forward_gpu(&input, Some(&mut cache), 0, 0, None, None);
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "10 in {:?} seconds (avg {:?} ms)",
        elapsed,
        elapsed / 10.0 * 1000.0
    );
}
