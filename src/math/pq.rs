//! Product Quantization implementation for LittleVector
//! 
//! This module contains the core PQ logic: splitting vectors,
//! training centroids, encoding documents, and performing searches.
use crate::math::distance::euclidean_distance;
/// Split a vector into equally-sized subvectors for Product Quantization
/// 
/// This is the first step of PQ: take a high-dimensional vector and
/// divide it into m subvectors that can be quantized independently.
/// 
/// # Arguments
/// * `vector` - The input vector to split
/// * `num_subvectors` - How many pieces to split into (m in our math)
/// 
/// # Returns
/// * Vec of subvectors, each containing d/m dimensions
/// 
/// # Panics
/// * If vector length isn't evenly divisible by num_subvectors
pub fn split_vector(vector: &[f32], num_subvectors: usize) -> Vec<Vec<f32>> {
    // Step 1: Check if split is possible
    assert_eq!(
        vector.len() % num_subvectors, 
        0, 
        "Vector length {} must be divisible by num_subvectors {}", 
        vector.len(), 
        num_subvectors
    );
    
    // Step 2: Calculate how big each subvector should be
    let subvector_size = vector.len() / num_subvectors;
    
    // Step 3: Create empty Vec to hold our subvectors
    let mut subvectors = Vec::new();
    
    // Step 4: Loop through and create each subvector
    for i in 0..num_subvectors {
        let start_idx = i * subvector_size;
        let end_idx = start_idx + subvector_size;
        
        // Step 5: Extract slice and convert to Vec
        let subvector = vector[start_idx..end_idx].to_vec();
        subvectors.push(subvector);
    }
    
    subvectors
}

/// Find the index of the closest centroid to a given subvector
/// 
/// This is used during PQ encoding: for each subvector, we find which
/// of the k centroids it's closest to, and store that index.
/// 
/// # Arguments
/// * `subvector` - The subvector to encode
/// * `centroids` - All available centroids for this subspace
/// 
/// # Returns
/// * Index of the closest centroid (0 to k-1)
pub fn find_closest_centroid(subvector: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best_index = 0;
    let mut best_distance = f32::INFINITY;
    
    for (index, centroid) in centroids.iter().enumerate() {
        let distance = euclidean_distance(subvector, centroid);
        
        if distance < best_distance {
            best_distance = distance;
            best_index = index;
        }
    }
    
    best_index
}

/// Encode a full vector into a PQ code using the given codebooks
/// 
/// This is the core of PQ: take a high-dimensional vector, split it into
/// subvectors, find the closest centroid for each, and return the indices.
/// 
/// # Arguments
/// * `vector` - The full vector to encode (e.g., 768 dimensions)
/// * `codebooks` - Pre-trained centroids for each subspace
/// 
/// # Returns
/// * PQ code as Vec<u8> (e.g., [42, 127, 203, 15, 88, 156, 7, 89])
pub fn encode_vector(vector: &[f32], codebooks: &[Vec<Vec<f32>>]) -> Vec<u8> {
    // Step 1: Split the vector into subvectors
    let num_subvectors = codebooks.len();
    let subvectors = split_vector(vector, num_subvectors);
    
    // Step 2: Find closest centroid for each subvector
    let mut pq_code = Vec::new();
    
    for (subvector, codebook) in subvectors.iter().zip(codebooks.iter()) {
        let centroid_index = find_closest_centroid(subvector, codebook);
        pq_code.push(centroid_index as u8); // Convert to u8 (0-255)
    }
    
    pq_code
}

/// Search for similar documents using asymmetric distance computation (ADC)
/// 
/// This is the core search algorithm: query stays full precision while
/// documents are represented by their compressed PQ codes.
/// 
/// # Arguments
/// * `query` - Full precision query vector
/// * `document_codes` - PQ codes for all documents
/// * `codebooks` - The trained centroids
/// * `top_k` - How many results to return
/// 
/// # Returns
/// * Vec of (document_index, distance) pairs, sorted by distance
pub fn search_pq(
    query: &[f32], 
    document_codes: &[Vec<u8>], 
    codebooks: &[Vec<Vec<f32>>], 
    top_k: usize
) -> Vec<(usize, f32)> {
    // Step 1: Split query into subvectors
    let num_subvectors = codebooks.len();
    let query_subvectors = split_vector(query, num_subvectors);
    
    // Step 2: Pre-compute distance tables
    let mut distance_tables = Vec::new();
    
    for (_subspace_idx, (query_subvector, codebook)) in 
        query_subvectors.iter().zip(codebooks.iter()).enumerate() {
        
        let mut distances_to_centroids = Vec::new();
        
        // Calculate distance from query subvector to ALL centroids in this codebook
        for centroid in codebook.iter() {
            let distance = euclidean_distance(query_subvector, centroid);
            distances_to_centroids.push(distance);
        }
        
        distance_tables.push(distances_to_centroids);
    }
    
    // Step 3: Score all documents using table lookups
    let mut results = Vec::new();
    
    for (doc_idx, pq_code) in document_codes.iter().enumerate() {
        let mut total_distance = 0.0;
        
        // Sum up distances from each subspace
        for (_subspace_idx, &centroid_idx) in pq_code.iter().enumerate() {
            total_distance += distance_tables[_subspace_idx][centroid_idx as usize];
        }
        
        results.push((doc_idx, total_distance));
    }
    
    // Step 4: Sort by distance and return top_k
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    results.truncate(top_k);
    
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_vector_basic() {
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let subvectors = split_vector(&vector, 4);
        
        assert_eq!(subvectors.len(), 4);
        assert_eq!(subvectors[0], vec![1.0, 2.0]);
        assert_eq!(subvectors[1], vec![3.0, 4.0]);
        assert_eq!(subvectors[2], vec![5.0, 6.0]);
        assert_eq!(subvectors[3], vec![7.0, 8.0]);
    }
}

#[test]
fn test_find_closest_centroid() {
    // Create some simple centroids
    let centroids = vec![
        vec![0.0, 0.0],  // Centroid 0 at origin
        vec![1.0, 1.0],  // Centroid 1 
        vec![5.0, 5.0],  // Centroid 2 far away
    ];
    
    // Test subvector close to centroid 1
    let subvector = vec![0.9, 1.1];
    let closest = find_closest_centroid(&subvector, &centroids);
    assert_eq!(closest, 1); // Should pick centroid 1
    
    // Test subvector close to origin
    let subvector2 = vec![0.1, 0.1];
    let closest2 = find_closest_centroid(&subvector2, &centroids);
    assert_eq!(closest2, 0); // Should pick centroid 0
}

#[test]
fn test_encode_vector() {
    // Create simple 4D vector and 2 subspaces
    let vector = vec![1.0, 2.0, 10.0, 11.0];
    
    // Create codebooks: 2 subspaces, 2 centroids each
    let codebooks = vec![
        // Codebook for first subspace (dims 0-1)
        vec![
            vec![0.0, 0.0],   // Centroid 0
            vec![1.0, 2.0],   // Centroid 1 - should match [1.0, 2.0]
        ],
        // Codebook for second subspace (dims 2-3) 
        vec![
            vec![0.0, 0.0],   // Centroid 0
            vec![10.0, 11.0], // Centroid 1 - should match [10.0, 11.0]
        ],
    ];
    
    let pq_code = encode_vector(&vector, &codebooks);
    assert_eq!(pq_code, vec![1, 1]); // Both subvectors match centroid 1
}

#[test]
fn test_search_pq() {
    // Create a simple setup
    let query = vec![1.0, 2.0, 10.0, 11.0];
    
    // Document codes
    let document_codes = vec![
        vec![1, 1], // Perfect match for our query
        vec![0, 0], // Should be farther
        vec![1, 0], // Mixed
    ];
    
    // Same codebooks as before
    let codebooks = vec![
        vec![
            vec![0.0, 0.0],   // Centroid 0
            vec![1.0, 2.0],   // Centroid 1
        ],
        vec![
            vec![0.0, 0.0],   // Centroid 0
            vec![10.0, 11.0], // Centroid 1
        ],
    ];
    
    let results = search_pq(&query, &document_codes, &codebooks, 2);
    
    // First result should be document 0 (perfect match)
    assert_eq!(results[0].0, 0);
    assert!(results[0].1 < 0.1); // Very small distance
}