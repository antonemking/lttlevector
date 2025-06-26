//! LittleVector: Educational Vector Database with Production Capabilities
//! 
//! A transparent implementation of Product Quantization and vector similarity search
//! that scales from educational demos to realistic workloads while maintaining
//! mathematical clarity.

pub mod math;
pub mod database;
pub mod demo;

// Re-export core functionality for clean public API
pub use database::{LittleVector, Document, SearchResult, SearchResults};
pub use math::distance::euclidean_distance;
pub use math::pq::{split_vector, encode_vector, search_pq};
pub use math::clustering::{CentroidTrainer, ConvergenceMetrics};

// For educational/demo purposes
pub use demo::{generate_sample_data, run_educational_demo, run_performance_demo};

/// Standard embedding dimensions supported
pub const DIMENSIONS_768: usize = 768;
pub const DIMENSIONS_384: usize = 384;
pub const DIMENSIONS_1536: usize = 1536;

/// Configuration presets for common use cases
pub mod presets {
    use crate::database::DatabaseConfig;

    /// Configuration optimized for learning and small datasets
    pub fn educational() -> DatabaseConfig {
        DatabaseConfig {
            num_centroids: 64,
            max_training_samples: 1000,
            enable_parallel_training: false,
            enable_progress_logging: true,
        }
    }

    /// Configuration for realistic workloads with good performance
    pub fn production() -> DatabaseConfig {
        DatabaseConfig {
            num_centroids: 256,
            max_training_samples: 50000,
            enable_parallel_training: true,
            enable_progress_logging: false,
        }
    }

    /// Configuration for maximum compression with acceptable search quality
    pub fn high_compression() -> DatabaseConfig {
        DatabaseConfig {
            num_centroids: 128,
            max_training_samples: 25000,
            enable_parallel_training: true,
            enable_progress_logging: false,
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_end_to_end_workflow() {
        // Create sample documents
        let docs = demo::generate_sample_data(100, Some(42));
        
        // Initialize database with production config
        let mut db = LittleVector::with_config(presets::production());
        
        // Train on subset
        let training_result = db.train(&docs[..50]).unwrap();
        assert!(training_result.success);
        
        // Add remaining documents
        let compression_result = db.add_documents(&docs[50..]).unwrap();
        assert!(compression_result.compression_ratio > 100.0);
        
        // Search
        let results = db.search(&docs[0].embedding, 5).unwrap();
        assert_eq!(results.documents.len(), 5);
        assert!(results.search_time_ms < 10.0); // Should be very fast
    }
}