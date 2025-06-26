// src/database.rs
//! Core vector database implementation
//! 
//! This module contains the main LittleVector database that handles both
//! educational workloads and realistic production scenarios seamlessly.

use crate::math::{distance::euclidean_distance, clustering::CentroidTrainer};
use std::collections::HashMap;
use std::time::Instant;
use rand::prelude::*;

// ================================
// Public Types and Interfaces
// ================================

pub type VectorId = String;
pub type Embedding = Vec<f32>;

#[derive(Clone, Debug)]
pub struct Document {
    pub id: VectorId,
    pub embedding: Embedding,
    pub metadata: HashMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub id: VectorId,
    pub distance: f32,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug)]
pub struct SearchResults {
    pub documents: Vec<SearchResult>,
    pub search_time_ms: f32,
    pub total_candidates: usize,
}

#[derive(Debug)]
pub struct TrainingResult {
    pub success: bool,
    pub training_time_ms: f32,
    pub convergence_iterations: Vec<usize>,
    pub memory_usage_mb: f32,
}

#[derive(Debug)]
pub struct CompressionResult {
    pub documents_added: usize,
    pub compression_ratio: f32,
    pub original_size_mb: f32,
    pub compressed_size_mb: f32,
    pub processing_time_ms: f32,
}

#[derive(Clone, Debug)]
pub struct DatabaseConfig {
    pub num_centroids: usize,
    pub max_training_samples: usize,
    pub enable_parallel_training: bool,
    pub enable_progress_logging: bool,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            num_centroids: 256,
            max_training_samples: 10000,
            enable_parallel_training: true,
            enable_progress_logging: true,
        }
    }
}

// ================================
// Core Database Implementation
// ================================

pub struct LittleVector {
    // Configuration
    config: DatabaseConfig,
    
    // Database state
    documents: Vec<CompressedDocument>,
    codebooks: Vec<Vec<SubVector>>,
    
    // Metadata
    vector_dimension: Option<usize>,
    num_subspaces: usize,
    is_trained: bool,
}

// Internal types
type SubVector = Vec<f32>;
type PQCode = Vec<u8>;

#[derive(Clone, Debug)]
struct CompressedDocument {
    id: VectorId,
    pq_code: PQCode,
    metadata: HashMap<String, String>,
}

impl LittleVector {
    /// Create a new database with default configuration
    pub fn new() -> Self {
        Self::with_config(DatabaseConfig::default())
    }

    /// Create a new database with custom configuration
    pub fn with_config(config: DatabaseConfig) -> Self {
        Self {
            config,
            documents: Vec::new(),
            codebooks: Vec::new(),
            vector_dimension: None,
            num_subspaces: 8, // Fixed for now, could be configurable
            is_trained: false,
        }
    }

    /// Train the database on a collection of documents
    pub fn train(&mut self, training_docs: &[Document]) -> Result<TrainingResult, String> {
        if training_docs.is_empty() {
            return Err("Cannot train on empty dataset".to_string());
        }

        let start_time = Instant::now();
        
        // Validate and determine vector dimension
        let dimension = self.validate_and_set_dimension(training_docs)?;
        
        if self.config.enable_progress_logging {
            println!("🧠 Training LittleVector on {} documents ({}-dimensional)", 
                     training_docs.len(), dimension);
        }

        // Determine subspace dimension
        if dimension % self.num_subspaces != 0 {
            return Err(format!("Vector dimension {} must be divisible by number of subspaces {}", 
                             dimension, self.num_subspaces));
        }
        let subspace_dim = dimension / self.num_subspaces;

        // Sample training data if too large
        let effective_training_docs = self.sample_training_data(training_docs);
        
        // Split vectors into subspaces
        let subvector_groups = self.split_training_vectors(&effective_training_docs, subspace_dim);
        
        // Train centroids for each subspace
        let mut convergence_iterations = Vec::new();
        self.codebooks = Vec::with_capacity(self.num_subspaces);
        
        for (subspace_idx, subvectors) in subvector_groups.iter().enumerate() {
            if self.config.enable_progress_logging {
                println!("  Training subspace {} ({} vectors)", subspace_idx, subvectors.len());
            }
            
            let mut trainer = CentroidTrainer::new(self.config.num_centroids);
            let centroids = trainer.train_centroids(subvectors);
            
            // Track convergence (if trainer exposes this info)
            convergence_iterations.push(trainer.convergence_history.len());
            
            self.codebooks.push(centroids);
        }

        self.is_trained = true;
        let training_time = start_time.elapsed();
        
        let result = TrainingResult {
            success: true,
            training_time_ms: training_time.as_secs_f32() * 1000.0,
            convergence_iterations,
            memory_usage_mb: self.calculate_memory_usage(),
        };

        if self.config.enable_progress_logging {
            println!("✅ Training complete in {:.1}ms", result.training_time_ms);
        }

        Ok(result)
    }

    /// Add documents to the database (must be trained first)
    pub fn add_documents(&mut self, docs: &[Document]) -> Result<CompressionResult, String> {
        if !self.is_trained {
            return Err("Database must be trained before adding documents".to_string());
        }

        if docs.is_empty() {
            return Ok(CompressionResult {
                documents_added: 0,
                compression_ratio: 0.0,
                original_size_mb: 0.0,
                compressed_size_mb: 0.0,
                processing_time_ms: 0.0,
            });
        }

        let start_time = Instant::now();
        let dimension = self.vector_dimension.unwrap();
        
        // Validate dimensions
        for (idx, doc) in docs.iter().enumerate() {
            if doc.embedding.len() != dimension {
                return Err(format!("Document {} has {} dimensions, expected {}", 
                                 idx, doc.embedding.len(), dimension));
            }
        }

        let original_size = docs.len() * dimension * std::mem::size_of::<f32>();

        // Compress each document
        for doc in docs {
            let pq_code = self.encode_to_pq(&doc.embedding);
            let compressed_doc = CompressedDocument {
                id: doc.id.clone(),
                pq_code,
                metadata: doc.metadata.clone(),
            };
            self.documents.push(compressed_doc);
        }

        let compressed_size = docs.len() * self.num_subspaces; // bytes
        let processing_time = start_time.elapsed();

        Ok(CompressionResult {
            documents_added: docs.len(),
            compression_ratio: original_size as f32 / compressed_size as f32,
            original_size_mb: original_size as f32 / 1_048_576.0,
            compressed_size_mb: compressed_size as f32 / 1_048_576.0,
            processing_time_ms: processing_time.as_secs_f32() * 1000.0,
        })
    }

    /// Search for similar documents
    pub fn search(&self, query: &Embedding, top_k: usize) -> Result<SearchResults, String> {
        if !self.is_trained {
            return Err("Database must be trained before searching".to_string());
        }

        if self.documents.is_empty() {
            return Ok(SearchResults {
                documents: Vec::new(),
                search_time_ms: 0.0,
                total_candidates: 0,
            });
        }

        let dimension = self.vector_dimension.unwrap();
        if query.len() != dimension {
            return Err(format!("Query has {} dimensions, expected {}", query.len(), dimension));
        }

        let start_time = Instant::now();

        // Split query into subvectors
        let query_subvectors = self.split_vector(query);

        // Pre-compute distance tables for asymmetric search
        let distance_tables = self.compute_distance_tables(&query_subvectors);

        // Score all documents
        let mut scored_results: Vec<(usize, f32)> = self.documents
            .iter()
            .enumerate()
            .map(|(idx, doc)| {
                let distance = self.compute_pq_distance(&doc.pq_code, &distance_tables);
                (idx, distance)
            })
            .collect();

        // Sort and take top-k
        scored_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let top_results = scored_results.into_iter().take(top_k).collect::<Vec<_>>();

        let search_time = start_time.elapsed();

        // Convert to result format
        let documents: Vec<SearchResult> = top_results
            .into_iter()
            .map(|(idx, distance)| SearchResult {
                id: self.documents[idx].id.clone(),
                distance,
                metadata: self.documents[idx].metadata.clone(),
            })
            .collect();

        Ok(SearchResults {
            documents,
            search_time_ms: search_time.as_secs_f32() * 1000.0,
            total_candidates: self.documents.len(),
        })
    }

    /// Get database statistics
    pub fn stats(&self) -> DatabaseStats {
        DatabaseStats {
            is_trained: self.is_trained,
            num_documents: self.documents.len(),
            vector_dimension: self.vector_dimension,
            num_subspaces: self.num_subspaces,
            num_centroids_per_subspace: self.config.num_centroids,
            memory_usage_mb: self.calculate_memory_usage(),
            config: self.config.clone(),
        }
    }

    // ================================
    // Internal Implementation
    // ================================

    fn validate_and_set_dimension(&mut self, docs: &[Document]) -> Result<usize, String> {
        if docs.is_empty() {
            return Err("No documents provided".to_string());
        }

        let dimension = docs[0].embedding.len();
        if dimension == 0 {
            return Err("Documents cannot have zero dimensions".to_string());
        }

        // Validate all documents have same dimension
        for (idx, doc) in docs.iter().enumerate() {
            if doc.embedding.len() != dimension {
                return Err(format!("Document {} has {} dimensions, expected {}", 
                                 idx, doc.embedding.len(), dimension));
            }
        }

        self.vector_dimension = Some(dimension);
        Ok(dimension)
    }

    fn sample_training_data<'a>(&self, docs: &'a [Document]) -> Vec<&'a Document> {
        if docs.len() <= self.config.max_training_samples {
            return docs.iter().collect();
        }

        if self.config.enable_progress_logging {
            println!("  Sampling {} documents from {} for training", 
                     self.config.max_training_samples, docs.len());
        }

        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..docs.len()).collect();
        indices.shuffle(&mut rng);
        
        indices.iter()
            .take(self.config.max_training_samples)
            .map(|&idx| &docs[idx])
            .collect()
    }

    fn split_training_vectors(&self, docs: &[&Document], subspace_dim: usize) -> Vec<Vec<SubVector>> {
        let mut subvector_groups = vec![Vec::new(); self.num_subspaces];
        
        for doc in docs {
            let subvectors = self.split_vector(&doc.embedding);
            for (idx, subvector) in subvectors.into_iter().enumerate() {
                subvector_groups[idx].push(subvector);
            }
        }
        
        subvector_groups
    }

    fn split_vector(&self, vector: &Embedding) -> Vec<SubVector> {
        let dimension = vector.len();
        let subspace_dim = dimension / self.num_subspaces;
        let mut subvectors = Vec::with_capacity(self.num_subspaces);
        
        for i in 0..self.num_subspaces {
            let start_idx = i * subspace_dim;
            let end_idx = start_idx + subspace_dim;
            let subvector = vector[start_idx..end_idx].to_vec();
            subvectors.push(subvector);
        }
        
        subvectors
    }

    fn encode_to_pq(&self, vector: &Embedding) -> PQCode {
        let subvectors = self.split_vector(vector);
        let mut pq_code = Vec::with_capacity(self.num_subspaces);
        
        for (subspace_idx, subvector) in subvectors.iter().enumerate() {
            let codebook = &self.codebooks[subspace_idx];
            let nearest_idx = self.find_nearest_centroid(subvector, codebook);
            pq_code.push(nearest_idx as u8);
        }
        
        pq_code
    }

    fn find_nearest_centroid(&self, subvector: &SubVector, codebook: &[SubVector]) -> usize {
        codebook.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                euclidean_distance(subvector, a)
                    .partial_cmp(&euclidean_distance(subvector, b))
                    .unwrap()
            })
            .unwrap()
            .0
    }

    fn compute_distance_tables(&self, query_subvectors: &[SubVector]) -> Vec<Vec<f32>> {
        query_subvectors.iter()
            .zip(self.codebooks.iter())
            .map(|(query_subvec, codebook)| {
                codebook.iter()
                    .map(|centroid| euclidean_distance(query_subvec, centroid))
                    .collect()
            })
            .collect()
    }

    fn compute_pq_distance(&self, pq_code: &PQCode, distance_tables: &[Vec<f32>]) -> f32 {
        pq_code.iter()
            .enumerate()
            .map(|(subspace_idx, &centroid_idx)| {
                distance_tables[subspace_idx][centroid_idx as usize]
            })
            .sum()
    }

    fn calculate_memory_usage(&self) -> f32 {
        let codebooks_size = if !self.codebooks.is_empty() {
            self.codebooks.len() * self.config.num_centroids * 
            (self.vector_dimension.unwrap_or(0) / self.num_subspaces) * 
            std::mem::size_of::<f32>()
        } else {
            0
        };
        
        let documents_size = self.documents.len() * self.num_subspaces;
        
        (codebooks_size + documents_size) as f32 / 1_048_576.0
    }
}

#[derive(Debug)]
pub struct DatabaseStats {
    pub is_trained: bool,
    pub num_documents: usize,
    pub vector_dimension: Option<usize>,
    pub num_subspaces: usize,
    pub num_centroids_per_subspace: usize,
    pub memory_usage_mb: f32,
    pub config: DatabaseConfig,
}

impl Default for LittleVector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_documents(count: usize, dim: usize) -> Vec<Document> {
        let mut docs = Vec::new();
        for i in 0..count {
            let mut metadata = HashMap::new();
            metadata.insert("test".to_string(), format!("doc_{}", i));
            
            docs.push(Document {
                id: format!("doc_{}", i),
                embedding: vec![i as f32; dim],
                metadata,
            });
        }
        docs
    }

    #[test]
    fn test_database_workflow() {
        let docs = create_test_documents(100, 768);
        let mut db = LittleVector::new();
        
        // Train
        let training_result = db.train(&docs[..50]).unwrap();
        assert!(training_result.success);
        
        // Add documents
        let compression_result = db.add_documents(&docs[50..]).unwrap();
        assert_eq!(compression_result.documents_added, 50);
        
        // Search
        let results = db.search(&docs[0].embedding, 5).unwrap();
        assert_eq!(results.documents.len(), 5);
    }

    #[test]
    fn test_different_configurations() {
        let docs = create_test_documents(50, 384);
        
        // Test educational config
        let mut db_edu = LittleVector::with_config(crate::presets::educational());
        db_edu.train(&docs[..25]).unwrap();
        db_edu.add_documents(&docs[25..]).unwrap();
        
        // Test production config
        let mut db_prod = LittleVector::with_config(crate::presets::production());
        db_prod.train(&docs[..25]).unwrap();
        db_prod.add_documents(&docs[25..]).unwrap();
        
        // Both should work but may have different performance characteristics
        let results_edu = db_edu.search(&docs[0].embedding, 3).unwrap();
        let results_prod = db_prod.search(&docs[0].embedding, 3).unwrap();
        
        assert_eq!(results_edu.documents.len(), 3);
        assert_eq!(results_prod.documents.len(), 3);
    }
}