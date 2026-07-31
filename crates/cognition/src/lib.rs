pub mod hypervector;
pub mod hnsw;
pub mod memory;
pub mod hd;
pub mod encoding;
pub mod features;
pub mod reasoning;
pub mod routing;
pub mod attention;

pub use hypervector::HyperVector;
pub use hnsw::BitHNSW;
pub use memory::SparseAssociativeMemory;
pub use encoding::{encode_activation_direct, RandomIndexCodebook};
pub use hd::{bundle, bind, permute, encode_sequence};
pub use features::FeatureExtractor;
pub use reasoning::ReasoningUnit;
pub use routing::Router;
pub use attention::SparseAttention;
