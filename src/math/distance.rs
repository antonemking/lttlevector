//! Distance calculation functions for vector similarity
//! 
//! This module provides the mathematical foundation for LittleVector's
//! similarity calculations.

/// Calculate Euclidean distance between two vectors
/// 
/// This is the foundation of Product Quantization - we'll use this
/// to compare query subvectors with centroids during search.
/// 
/// # Arguments
/// * `vec1` - First vector slice
/// * `vec2` - Second vector slice
/// 
/// # Returns
/// * Euclidean distance as f32
/// 
/// # Panics
/// * If vectors have different lengths
pub fn euclidean_distance(vec1: &[f32], vec2: &[f32]) -> f32 {
    assert_eq!(vec1.len(), vec2.len(), "Vectors must have same length");
    
    let mut sum_squared_diff = 0.0;
    
    for i in 0..vec1.len() {
        let diff = vec1[i] - vec2[i];
        sum_squared_diff += diff * diff;
    }
    
    sum_squared_diff.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance_basic() {
        let vec_a = vec![0.0, 0.0];
        let vec_b = vec![3.0, 4.0];
        
        let distance = euclidean_distance(&vec_a, &vec_b);
        assert!((distance - 5.0).abs() < 1e-6); // Should be 5.0
    }
    
    #[test]
    fn test_euclidean_distance_identical() {
        let vec_a = vec![1.0, 2.0, 3.0];
        let vec_b = vec![1.0, 2.0, 3.0];
        
        let distance = euclidean_distance(&vec_a, &vec_b);
        assert!(distance < 1e-6); // Should be ~0.0
    }
}