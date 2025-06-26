pub mod math;

pub use math::distance::euclidean_distance;
pub use math::pq::{split_vector, find_closest_centroid, encode_vector, search_pq};
pub use math::clustering::{CentroidTrainer, ConvergenceMetrics};