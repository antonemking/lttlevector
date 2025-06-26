//! Demo utilities and test data generation
//! 
//! This module provides functions for generating test data and running
//! demonstration workflows that showcase LittleVector's capabilities.

use crate::db::{Document, LittleVector};
use crate::presets;
use std::collections::HashMap;
use std::time::Instant;
use rand::prelude::*;

/// Generate sample data for testing and demos
pub fn generate_sample_data(count: usize, seed: Option<u64>) -> Vec<Document> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    generate_embeddings_with_clusters(count, 768, 10, &mut rng)
}

/// Generate embeddings with realistic clustering structure
pub fn generate_embeddings_with_clusters(
    count: usize, 
    dimension: usize, 
    num_clusters: usize,
    rng: &mut StdRng
) -> Vec<Document> {
    let mut docs = Vec::with_capacity(count);
    
    for i in 0..count {
        let cluster_id = i % num_clusters;
        let mut embedding = vec![0.0f32; dimension];
        
        // Create cluster-specific patterns
        let base_value = (cluster_id as f32) * 0.1;
        
        for j in 0..dimension {
            // Base cluster signal
            embedding[j] = base_value + rng.gen_range(-0.05..0.05);
            
            // Add cluster-specific features in different parts of the vector
            let cluster_region_start = (cluster_id * dimension / num_clusters).min(dimension - 50);
            let cluster_region_end = ((cluster_id + 1) * dimension / num_clusters).min(dimension);
            
            if j >= cluster_region_start && j < cluster_region_end {
                embedding[j] += rng.gen_range(0.3..0.8);
            }
            
            // Add small amount of global noise
            embedding[j] += rng.gen_range(-0.02..0.02);
        }
        
        let mut metadata = HashMap::new();
        metadata.insert("cluster".to_string(), format!("cluster_{}", cluster_id));
        metadata.insert("category".to_string(), format!("category_{}", cluster_id % 3));
        metadata.insert("source".to_string(), "generated".to_string());
        
        docs.push(Document {
            id: format!("doc_{:06}", i),
            embedding,
            metadata,
        });
    }
    
    docs
}

/// Generate simple educational data (small dimensions for easy understanding)
pub fn generate_educational_data() -> Vec<Document> {
    let mut docs = Vec::new();
    
    // Create some simple 8-dimensional vectors with clear patterns
    // But we need to make them 16-dimensional so they're divisible by 8 subspaces
    let patterns = vec![
        (vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], "pattern_a"),
        (vec![5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0], "pattern_b"),
        (vec![2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0], "pattern_c"),
    ];
    
    for (i, (base_pattern, pattern_type)) in patterns.iter().enumerate() {
        for j in 0..15 { // 15 documents per pattern
            let mut embedding = base_pattern.clone();
            
            // Add small random variations
            let mut rng = thread_rng();
            for val in embedding.iter_mut() {
                *val += rng.gen_range(-0.2..0.2);
            }
            
            let mut metadata = HashMap::new();
            metadata.insert("pattern".to_string(), pattern_type.to_string());
            metadata.insert("group".to_string(), format!("group_{}", i));
            
            docs.push(Document {
                id: format!("edu_doc_{}_{}", i, j),
                embedding,
                metadata,
            });
        }
    }
    
    docs
}

/// Run educational demo with small, interpretable data
pub fn run_educational_demo() -> Result<(), String> {
    println!("📚 Educational Demo: Understanding Product Quantization");
    println!("======================================================");
    
    // Generate simple data
    let docs = generate_educational_data();
    println!("📊 Generated {} educational documents (16-dimensional)", docs.len());
    
    // Use educational configuration
    let mut db = LittleVector::with_config(presets::educational());
    
    // Train on half the data
    println!("\n🧠 Training phase:");
    let training_result = db.train(&docs[..30])?;
    println!("✅ Training completed in {:.1}ms", training_result.training_time_ms);
    println!("📈 Convergence iterations: {:?}", training_result.convergence_iterations);
    
    // Add remaining documents
    println!("\n🗜️  Compression phase:");
    let compression_result = db.add_documents(&docs[30..])?;
    println!("✅ Compressed {} documents", compression_result.documents_added);
    println!("📉 Compression ratio: {:.1}x", compression_result.compression_ratio);
    println!("💾 Size: {:.3}MB → {:.3}MB", 
             compression_result.original_size_mb, 
             compression_result.compressed_size_mb);
    
    // Demo searches
    println!("\n🔍 Search phase:");
    for i in 0..3 {
        let query_doc = &docs[i * 10];
        let results = db.search(&query_doc.embedding, 5)?;
        
        println!("Query {}: {} (pattern: {})", 
                 i + 1, 
                 query_doc.id,
                 query_doc.metadata.get("pattern").map_or("unknown", |v| v.as_str()));
        
        for (rank, result) in results.documents.iter().enumerate() {
            let result_cluster = result.metadata.get("cluster").map_or("unknown", |v| v.as_str());
            println!("  {}. {} (distance: {:.3}, pattern: {})", 
                     rank + 1, 
                     result.id, 
                     result.distance,
                     result_cluster);
        }
        println!();
    }
    
    let stats = db.stats();
    println!("📊 Final statistics:");
    println!("  Documents: {}", stats.num_documents);
    println!("  Dimensions: {:?}", stats.vector_dimension);
    println!("  Memory: {:.2}MB", stats.memory_usage_mb);
    
    Ok(())
}

/// Run performance demo with realistic data sizes
pub fn run_performance_demo() -> Result<(), String> {
    println!("🎯 Performance Demo: Realistic Scale Workload");
    println!("==============================================");
    
    // Generate realistic dataset
    let num_docs = 2000;
    let docs = generate_sample_data(num_docs, Some(42));
    println!("📊 Generated {} documents (768-dimensional)", docs.len());
    
    // Use production configuration
    let mut db = LittleVector::with_config(presets::production());
    
    // Benchmark training
    println!("\n🧠 Training phase:");
    let train_start = Instant::now();
    let training_result = db.train(&docs[..500])?;
    let _train_time = train_start.elapsed();
    
    println!("✅ Trained on 500 documents in {:.1}ms", training_result.training_time_ms);
    println!("📈 Average convergence: {:.1} iterations", 
             training_result.convergence_iterations.iter().sum::<usize>() as f32 / 
             training_result.convergence_iterations.len() as f32);
    
    // Benchmark compression
    println!("\n🗜️  Compression phase:");
    let compress_start = Instant::now();
    let compression_result = db.add_documents(&docs[500..])?;
    let _compress_time = compress_start.elapsed();
    
    println!("✅ Compressed {} documents in {:.1}ms", 
             compression_result.documents_added, compression_result.processing_time_ms);
    println!("📉 Compression ratio: {:.1}x", compression_result.compression_ratio);
    println!("💾 Memory saved: {:.1}MB → {:.1}MB", 
             compression_result.original_size_mb, 
             compression_result.compressed_size_mb);
    
    // Benchmark search performance
    println!("\n🔍 Search performance:");
    let search_queries = vec![
        &docs[0].embedding,
        &docs[200].embedding, 
        &docs[500].embedding,
        &docs[1000].embedding,
    ];
    
    let mut total_search_time = 0.0;
    for (i, query) in search_queries.iter().enumerate() {
        let search_start = Instant::now();
        let results = db.search(query, 10)?;
        let _search_time = search_start.elapsed();
        
        total_search_time += results.search_time_ms;
        
        println!("  Query {}: {:.3}ms ({} candidates)", 
                 i + 1, results.search_time_ms, results.total_candidates);
        
        // Show top 3 results
        for (rank, result) in results.documents.iter().take(3).enumerate() {
            println!("    {}. {} (distance: {:.3})", rank + 1, result.id, result.distance);
        }
    }
    
    let avg_search_time = total_search_time / search_queries.len() as f32;
    println!("📊 Average search time: {:.3}ms", avg_search_time);
    
    // Final statistics
    let stats = db.stats();
    println!("\n📈 Performance Summary:");
    println!("================================");
    println!("Training throughput: {:.0} docs/sec", 
             500.0 / (training_result.training_time_ms / 1000.0));
    println!("Compression throughput: {:.0} docs/sec", 
             compression_result.documents_added as f32 / (compression_result.processing_time_ms / 1000.0));
    println!("Search throughput: {:.0} queries/sec", 
             1000.0 / avg_search_time);
    println!("Memory efficiency: {:.1}x compression", compression_result.compression_ratio);
    println!("Total memory usage: {:.2}MB", stats.memory_usage_mb);
    
    // Demonstrate different search patterns
    println!("\n🎯 Search Quality Analysis:");
    demonstrate_search_quality(&db, &docs)?;
    
    Ok(())
}

/// Demonstrate search quality with different query types
fn demonstrate_search_quality(db: &LittleVector, docs: &[Document]) -> Result<(), String> {
    // Test 1: Exact match (should return the document itself as top result)
    println!("\n1. Exact Match Test:");
    let exact_query = &docs[500].embedding; // Document in the database
    let exact_results = db.search(exact_query, 5)?;
    
    println!("   Query document: {}", docs[500].id);
    println!("   Top result: {} (distance: {:.6})", 
             exact_results.documents[0].id, exact_results.documents[0].distance);
    
    if exact_results.documents[0].distance < 0.1 {
        println!("   ✅ Perfect match found (distance < 0.1)");
    } else {
        println!("   ⚠️  Inexact match due to quantization (distance: {:.3})", 
                 exact_results.documents[0].distance);
    }
    
    // Test 2: Cluster coherence (documents from same cluster should be similar)
    println!("\n2. Cluster Coherence Test:");
    let cluster_query = &docs[100].embedding; // Pick a document from cluster
    let cluster_results = db.search(cluster_query, 10)?;
    
    let query_cluster = docs[100].metadata.get("cluster").unwrap();
    let mut same_cluster_count = 0;
    
    println!("   Query cluster: {}", query_cluster);
    println!("   Top 5 results:");
    
    for (rank, result) in cluster_results.documents.iter().take(5).enumerate() {
        let result_cluster = result.metadata.get("cluster").map_or("unknown", |v| v.as_str());
        if result_cluster == query_cluster {
            same_cluster_count += 1;
        }
        println!("     {}. {} (cluster: {}, distance: {:.3})", 
                 rank + 1, result.id, result_cluster, result.distance);
    }
    
    let cluster_precision = same_cluster_count as f32 / 5.0;
    println!("   📊 Cluster precision@5: {:.1}% ({}/5 same cluster)", 
             cluster_precision * 100.0, same_cluster_count);
    
    // Test 3: Search speed vs accuracy tradeoff
    println!("\n3. Speed vs Accuracy Analysis:");
    let test_queries = docs.iter().step_by(100).take(10).collect::<Vec<_>>();
    
    let mut total_time = 0.0;
    let mut perfect_matches = 0;
    let mut good_matches = 0; // distance < 1.0
    
    for query_doc in test_queries {
        let results = db.search(&query_doc.embedding, 1)?;
        total_time += results.search_time_ms;
        
        if !results.documents.is_empty() {
            let top_distance = results.documents[0].distance;
            if top_distance < 0.1 {
                perfect_matches += 1;
            } else if top_distance < 1.0 {
                good_matches += 1;
            }
        }
    }
    
    let avg_time = total_time / 10.0; // 10 queries
    println!("   Average search time: {:.3}ms", avg_time);
    println!("   Perfect matches (d<0.1): {}/10", perfect_matches);
    println!("   Good matches (d<1.0): {}/10", good_matches);
    println!("   Search accuracy: {:.1}%", (perfect_matches + good_matches) as f32 * 10.0);
    
    Ok(())
}

/// Compare different configuration presets
pub fn run_configuration_comparison() -> Result<(), String> {
    println!("⚖️  Configuration Comparison Demo");
    println!("=================================");
    
    let docs = generate_sample_data(500, Some(123));
    let test_query = &docs[0].embedding;
    
    let configs = vec![
        ("Educational", presets::educational()),
        ("Production", presets::production()),
        ("High Compression", presets::high_compression()),
    ];
    
    println!("Testing {} documents (768D) with different configurations:\n", docs.len());
    
    for (name, config) in configs {
        println!("🔧 Testing {} Configuration:", name);
        println!("   Centroids: {}", config.num_centroids);
        println!("   Max training samples: {}", config.max_training_samples);
        println!("   Parallel training: {}", config.enable_parallel_training);
        
        let mut db = LittleVector::with_config(config);
        
        // Train and measure
        let training_result = db.train(&docs[..250])?;
        
        // Add documents and measure
        let compression_result = db.add_documents(&docs[250..])?;
        
        // Search and measure
        let search_results = db.search(test_query, 10)?;
        
        // Report results
        println!("   Results:");
        println!("     Training time: {:.1}ms", training_result.training_time_ms);
        println!("     Compression ratio: {:.1}x", compression_result.compression_ratio);
        println!("     Search time: {:.3}ms", search_results.search_time_ms);
        println!("     Memory usage: {:.2}MB", db.stats().memory_usage_mb);
        println!("     Top result distance: {:.3}", search_results.documents[0].distance);
        println!();
    }
    
    Ok(())
}

/// Generate embeddings that simulate real-world document embeddings
pub fn generate_document_embeddings(document_texts: &[&str]) -> Vec<Document> {
    let mut docs = Vec::new();
    let mut rng = thread_rng();
    
    for (i, text) in document_texts.iter().enumerate() {
        // Simulate document embedding by creating vector based on text characteristics
        let mut embedding = vec![0.0f32; 768];
        
        // Base embedding from text length and content
        let text_len_factor = (text.len() as f32).ln() * 0.1;
        let char_sum = text.chars().map(|c| c as u32).sum::<u32>() as f32 * 0.0001;
        
        for j in 0..768 {
            // Create patterns based on text characteristics
            embedding[j] = text_len_factor + char_sum + rng.gen_range(-0.1..0.1);
            
            // Add word-specific patterns (simulate semantic content)
            if text.contains("contract") && j < 100 {
                embedding[j] += 0.5;
            }
            if text.contains("policy") && j >= 100 && j < 200 {
                embedding[j] += 0.5;
            }
            if text.contains("agreement") && j >= 200 && j < 300 {
                embedding[j] += 0.5;
            }
            if text.contains("legal") && j >= 300 && j < 400 {
                embedding[j] += 0.3;
            }
        }
        
        let mut metadata = HashMap::new();
        metadata.insert("content_preview".to_string(), text.chars().take(50).collect());
        metadata.insert("doc_type".to_string(), "text".to_string());
        metadata.insert("length".to_string(), text.len().to_string());
        
        docs.push(Document {
            id: format!("text_doc_{}", i),
            embedding,
            metadata,
        });
    }
    
    docs
}

/// Run a demo with realistic document text
pub fn run_document_search_demo() -> Result<(), String> {
    println!("📄 Document Search Demo");
    println!("=======================");
    
    let document_texts = vec![
        "This employment contract establishes the terms and conditions of employment between the company and the employee.",
        "Privacy policy: This document outlines how we collect, use, and protect your personal information.",
        "Software license agreement: By using this software, you agree to the terms and conditions outlined herein.",
        "Terms of service: These terms govern your use of our website and services.",
        "Non-disclosure agreement: This agreement protects confidential information shared between parties.",
        "Rental agreement: This lease agreement establishes the terms for renting the property.",
        "Service contract: This contract outlines the services to be provided and payment terms.",
        "Partnership agreement: This document establishes the terms of the business partnership.",
        "Sales agreement: This contract covers the sale of goods and delivery terms.",
        "Consulting agreement: This agreement defines the scope of consulting services.",
    ];
    
    let docs = generate_document_embeddings(&document_texts);
    println!("Generated {} document embeddings", docs.len());
    
    let mut db = LittleVector::with_config(presets::production());
    
    // Train on all documents (small dataset)
    let training_result = db.train(&docs)?;
    println!("✅ Training completed in {:.1}ms", training_result.training_time_ms);
    
    // Demo semantic search
    println!("\n🔍 Semantic Search Examples:");
    
    // Create query embeddings for different concepts
    let query_texts = vec![
        "employment contract terms",
        "privacy and data protection",
        "legal agreement conditions",
    ];
    
    for query_text in query_texts {
        println!("\n📝 Query: \"{}\"", query_text);
        
        // Generate query embedding (same way as documents)
        let query_docs = generate_document_embeddings(&[query_text]);
        let query_embedding = &query_docs[0].embedding;
        
        let results = db.search(query_embedding, 3)?;
        
        println!("   Top matches:");
        for (rank, result) in results.documents.iter().enumerate() {
            let preview = result.metadata.get("content_preview").unwrap();
            println!("     {}. {} (distance: {:.3})", rank + 1, preview, result.distance);
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_data_generation() {
        let docs = generate_sample_data(100, Some(42));
        assert_eq!(docs.len(), 100);
        assert_eq!(docs[0].embedding.len(), 768);
        
        // Check that documents have cluster metadata
        assert!(docs[0].metadata.contains_key("cluster"));
    }

    #[test]
    fn test_educational_data() {
        let docs = generate_educational_data();
        assert_eq!(docs.len(), 45); // 3 patterns × 15 docs each
        assert_eq!(docs[0].embedding.len(), 16);
        
        // Check pattern metadata
        assert!(docs[0].metadata.get("pattern").unwrap().starts_with("pattern_"));
    }

    #[test]
    fn test_document_embeddings() {
        let texts = vec!["test document", "another test"];
        let docs = generate_document_embeddings(&texts);
        
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].embedding.len(), 768);
        assert!(docs[0].metadata.contains_key("content_preview"));
    }
}