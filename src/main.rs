use littlevector::{euclidean_distance, encode_vector, search_pq, CentroidTrainer};

fn main() {
    println!("🚀 LittleVector: Complete Product Quantization Demo with K-means Training");
    println!("{}", "=".repeat(70));
    
    // ========================================================================
    // STEP 1: Generate sample data for clustering demonstration
    // ========================================================================
    println!("\n📚 Step 1: Generate Sample Data");
    
    let training_data = generate_sample_vectors();
    println!("  Generated {} training vectors for centroid training", training_data.len());
    
    // Sample documents for search demo
    let documents = vec![
        ("contract_termination.txt", vec![1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0]),
        ("employment_agreement.txt", vec![1.1, 2.1, 3.1, 4.1, 8.0, 9.0, 10.0, 11.0]),
        ("privacy_policy.txt", vec![5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0]),
        ("terms_of_service.txt", vec![0.9, 1.9, 2.9, 3.9, 9.5, 10.5, 11.5, 12.5]),
    ];
    
    println!("  Sample documents for search:");
    for (name, vector) in &documents {
        println!("    📄 {}: {:?}", name, vector);
    }
    
    // ========================================================================
    // STEP 2: Train codebooks using k-means clustering
    // ========================================================================
    println!("\n🧠 Step 2: Train Codebooks with K-means");
    
    let num_subvectors = 2;
    let subvector_size = training_data[0].len() / num_subvectors;
    
    println!("  Splitting {} dimensional vectors into {} subvectors of {} dimensions", 
             training_data[0].len(), num_subvectors, subvector_size);
    
    // Split training data into subvectors for each subspace
    let mut subvector_groups = vec![Vec::new(); num_subvectors];
    for vector in &training_data {
        for subspace in 0..num_subvectors {
            let start = subspace * subvector_size;
            let end = start + subvector_size;
            let subvector = vector[start..end].to_vec();
            subvector_groups[subspace].push(subvector);
        }
    }
    
    // Train centroids for each subspace
    let mut codebooks = Vec::new();
    for (subspace_idx, subvectors) in subvector_groups.iter().enumerate() {
        println!("\n  🎯 Training centroids for subspace {} (dimensions {}-{})", 
                 subspace_idx, 
                 subspace_idx * subvector_size, 
                 (subspace_idx + 1) * subvector_size - 1);
        
        let mut trainer = CentroidTrainer::new(3); // 3 centroids per subspace
        let centroids = trainer.train_centroids(subvectors);
        
        println!("    ✅ Trained {} centroids for subspace {}", centroids.len(), subspace_idx);
        codebooks.push(centroids);
    }
    
    println!("\n  📊 Training Complete:");
    println!("    {} codebooks trained", codebooks.len());
    println!("    {} centroids per codebook", codebooks[0].len());
    
    // ========================================================================
    // STEP 3: Encode documents into PQ codes
    // ========================================================================
    println!("\n🔢 Step 3: Encode Documents to PQ Codes");
    
    let mut document_codes = Vec::new();
    
    for (name, vector) in &documents {
        let pq_code = encode_vector(vector, &codebooks);
        document_codes.push(pq_code.clone());
        
        println!("  📄 {}: {} bytes → {:?}", 
                 name, 
                 vector.len() * 4, // 4 bytes per f32
                 pq_code);
    }
    
    let original_size = documents.len() * 8 * 4; // 4 docs × 8 dims × 4 bytes
    let compressed_size = documents.len() * 2;    // 4 docs × 2 bytes
    println!("  💾 Compression: {} bytes → {} bytes ({:.1}x smaller)", 
             original_size, compressed_size, 
             original_size as f32 / compressed_size as f32);
    
    // ========================================================================
    // STEP 4: Search with a query
    // ========================================================================
    println!("\n🔍 Step 4: Search Demo");
    
    let query = vec![1.05, 2.05, 3.05, 4.05, 10.2, 11.2, 12.2, 13.2];
    println!("  🎯 Query vector: {:?}", query);
    
    // Perform PQ search
    let results = search_pq(&query, &document_codes, &codebooks, 3);
    
    println!("  📊 Search Results (top 3):");
    for (rank, (doc_idx, distance)) in results.iter().enumerate() {
        let doc_name = &documents[*doc_idx].0;
        println!("    {}. {} (distance: {:.3})", rank + 1, doc_name, distance);
    }
    
    // ========================================================================
    // STEP 5: Compare with exact search
    // ========================================================================
    println!("\n⚖️  Step 5: Accuracy Comparison");
    
    // Exact search for comparison
    let mut exact_results = Vec::new();
    for (idx, (_name, vector)) in documents.iter().enumerate() {
        let distance = euclidean_distance(&query, vector);
        exact_results.push((idx, distance));
    }
    exact_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    println!("  🎯 Exact search results:");
    for (rank, (doc_idx, distance)) in exact_results.iter().take(3).enumerate() {
        let doc_name = &documents[*doc_idx].0;
        println!("    {}. {} (distance: {:.3})", rank + 1, doc_name, distance);
    }
    
    // Check if top result matches
    let pq_top = results[0].0;
    let exact_top = exact_results[0].0;
    if pq_top == exact_top {
        println!("  ✅ PQ found the same top result as exact search!");
    } else {
        println!("  ⚠️  PQ top result differs from exact search (expected with compression)");
    }
    
    println!("\n🎉 Demo Complete! You've seen the complete PQ pipeline:");
    println!("   • K-means clustering for centroid training");
    println!("   • Convergence tracking and metrics");
    println!("   • Vector splitting and encoding");
    println!("   • Massive compression ({}x smaller)", 
             original_size / compressed_size);
    println!("   • Fast asymmetric search");
    println!("   • Accuracy comparison with exact search");
}

/// Generate sample vectors with natural clustering structure
/// 
/// Creates vectors that have clear cluster patterns so k-means
/// training will find meaningful centroids. In real applications,
/// these would be embeddings from your actual documents.
fn generate_sample_vectors() -> Vec<Vec<f32>> {
    use rand::prelude::*;
    let mut rng = thread_rng();
    let mut vectors = Vec::new();
    
    // Create 3 natural clusters in 8D space
    let cluster_centers = vec![
        vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],      // Cluster A: low values
        vec![8.0, 9.0, 10.0, 11.0, 8.0, 9.0, 10.0, 11.0],  // Cluster B: high values  
        vec![4.0, 5.0, 6.0, 7.0, 11.0, 12.0, 13.0, 14.0],  // Cluster C: mixed values
    ];
    
    // Generate points around each cluster center
    for center in &cluster_centers {
        for _ in 0..15 { // 15 points per cluster = 45 total training vectors
            let mut point = center.clone();
            for value in point.iter_mut() {
                *value += rng.gen_range(-0.8..0.8); // Add some noise
            }
            vectors.push(point);
        }
    }
    
    vectors
}