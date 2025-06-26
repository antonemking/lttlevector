//! K-means clustering implementation for Product Quantization centroid training
//! 
//! This module implements k-means clustering from scratch to train the centroids
//! that form our PQ codebooks. The implementation prioritizes mathematical
//! transparency over performance - every step is visible and measurable.
//!
//! Key design principles:
//! - Educational clarity: each function reveals its mathematical purpose
//! - Complete convergence tracking: measure and document the training process
//! - Reference implementation: clean baseline for understanding clustering

use rand::prelude::*;
use crate::math::distance::euclidean_distance;

/// Metrics captured during each k-means iteration
/// 
/// Tracks the mathematical progress of centroid training to understand
/// convergence behavior. Each metric reveals a different aspect of how
/// the clustering algorithm settles into stable configurations.
/// 
/// # Fields
/// * `iteration` - Current iteration number (1-indexed)
/// * `centroid_shift` - Average distance centroids moved this iteration
/// * `total_distortion` - Sum of squared distances from points to centroids
/// * `cluster_populations` - Number of points assigned to each cluster
/// * `stability_score` - Measure of cluster balance, 0.0 = unbalanced, 1.0 = perfectly balanced
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics {
    pub iteration: usize,
    pub centroid_shift: f32,
    pub total_distortion: f32,
    pub cluster_populations: Vec<usize>,
    pub stability_score: f32,
}

/// K-means trainer with full convergence documentation
/// 
/// Implements k-means clustering with k-means++ initialization and
/// comprehensive tracking of the training process. Designed to be
/// a reference implementation that teaches clustering fundamentals.
/// 
/// # Configuration
/// * `num_centroids` - Number of clusters to create
/// * `max_iterations` - Stop training after this many iterations
/// * `convergence_threshold` - Stop when centroid movement falls below this
/// * `convergence_history` - Complete record of training progress
pub struct CentroidTrainer {
    pub num_centroids: usize,
    pub max_iterations: usize,
    pub convergence_threshold: f32,
    pub convergence_history: Vec<ConvergenceMetrics>,
}

impl CentroidTrainer {
    /// Create a new k-means trainer
    /// 
    /// Sets up a trainer with standard parameters suitable for most
    /// educational demonstrations of clustering mathematics.
    /// 
    /// # Arguments
    /// * `num_centroids` - Number of clusters to find in the data
    /// 
    /// # Returns
    /// * Configured trainer ready for centroid training
    pub fn new(num_centroids: usize) -> Self {
        Self {
            num_centroids,
            max_iterations: 100,
            convergence_threshold: 1e-6,
            convergence_history: Vec::new(),
        }
    }

    /// Train centroids using k-means clustering
    /// 
    /// Runs the complete k-means algorithm with k-means++ initialization
    /// and full convergence tracking. This is the main method that
    /// coordinates the entire clustering process.
    /// 
    /// # Arguments
    /// * `data` - Vector of data points to cluster
    /// 
    /// # Returns
    /// * Vector of trained centroids
    /// 
    /// # Process
    /// 1. Initialize centroids using k-means++
    /// 2. Iteratively assign points and update centroids
    /// 3. Track convergence metrics each iteration
    /// 4. Stop when centroids stabilize or max iterations reached
    pub fn train_centroids(&mut self, data: &[Vec<f32>]) -> Vec<Vec<f32>> {
        println!("Training {} centroids on {} data points", self.num_centroids, data.len());
        
        if data.is_empty() {
            println!("Warning: No data provided for training");
            return vec![];
        }
        
        // Initialize centroids using k-means++
        let mut centroids = self.initialize_centroids_plus_plus(data);
        
        // Clear convergence history for this training session
        self.convergence_history.clear();
        
        // Main training loop
        for iteration in 0..self.max_iterations {
            // Assignment step: assign each point to nearest centroid
            let assignments = self.assign_points_to_centroids(data, &centroids);
            
            // Update step: move centroids to center of assigned points
            let new_centroids = self.update_centroids(data, &assignments);
            
            // Calculate convergence metrics
            let centroid_shift = self.calculate_centroid_shift(&centroids, &new_centroids);
            let total_distortion = self.calculate_total_distortion(data, &new_centroids, &assignments);
            let stability_score = self.calculate_stability_score(&assignments);
            let cluster_populations = self.count_cluster_populations(&assignments);
            
            let metrics = ConvergenceMetrics {
                iteration: iteration + 1,
                centroid_shift,
                total_distortion,
                cluster_populations,
                stability_score,
            };
            
            self.convergence_history.push(metrics.clone());
            
            // Print progress updates
            if iteration % 10 == 0 || centroid_shift < self.convergence_threshold {
                println!("  Iteration {}: shift={:.6}, distortion={:.2}, stability={:.3}", 
                    iteration + 1, centroid_shift, total_distortion, stability_score);
            }
            
            // Check for convergence
            if centroid_shift < self.convergence_threshold {
                println!("Converged after {} iterations (shift: {:.8})", iteration + 1, centroid_shift);
                break;
            }
            
            centroids = new_centroids;
        }
        
        self.print_convergence_summary();
        centroids
    }

    /// Initialize centroids using k-means++ algorithm
    /// 
    /// Places centroids with careful spacing across the data landscape.
    /// First centroid is random, subsequent centroids are placed with
    /// probability proportional to squared distance from nearest existing centroid.
    /// 
    /// # Arguments
    /// * `data` - Data points to cluster
    /// 
    /// # Returns
    /// * Vector of initial centroid positions
    fn initialize_centroids_plus_plus(&self, data: &[Vec<f32>]) -> Vec<Vec<f32>> {
        if data.is_empty() || self.num_centroids == 0 {
            return Vec::new();
        }

        let mut rng = thread_rng();
        let mut centroids = Vec::new();
        
        // First centroid chosen uniformly at random
        let first_idx = rng.gen_range(0..data.len());
        centroids.push(data[first_idx].clone());
        
        // Each subsequent centroid positioned with probability proportional
        // to squared distance from nearest existing centroid
        for _ in 1..self.num_centroids {
            let mut distances_squared = Vec::new();
            
            // Calculate squared distances to nearest centroid for each point
            for point in data.iter() {
                let min_distance = centroids.iter()
                    .map(|centroid| euclidean_distance(point, centroid))
                    .fold(f32::INFINITY, f32::min);
                distances_squared.push(min_distance * min_distance);
            }
            
            // Select next centroid using weighted probability
            let total: f32 = distances_squared.iter().sum();
            if total == 0.0 {
                // All points are identical, just add duplicates
                centroids.push(data[0].clone());
                continue;
            }
            
            let threshold = rng.gen::<f32>() * total;
            let mut cumulative = 0.0;
            
            for (idx, &dist_sq) in distances_squared.iter().enumerate() {
                cumulative += dist_sq;
                if cumulative >= threshold {
                    centroids.push(data[idx].clone());
                    break;
                }
            }
        }
        
        centroids
    }

    /// Assign each data point to its nearest centroid
    /// 
    /// This is the "assignment step" of k-means. Each point finds
    /// the centroid it's closest to using Euclidean distance.
    /// 
    /// # Arguments
    /// * `data` - Data points to assign
    /// * `centroids` - Current centroid positions
    /// 
    /// # Returns
    /// * Vector where assignments[i] is the centroid index for data[i]
    fn assign_points_to_centroids(&self, data: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<usize> {
        data.iter()
            .map(|point| {
                centroids.iter()
                    .enumerate()
                    .map(|(idx, centroid)| (idx, euclidean_distance(point, centroid)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Update centroids to the center of mass of assigned points
    /// 
    /// This is the "update step" of k-means. Each centroid moves to
    /// the average position of all points assigned to it.
    /// 
    /// # Arguments
    /// * `data` - All data points
    /// * `assignments` - Which centroid each point is assigned to
    /// 
    /// # Returns
    /// * Vector of updated centroid positions
    fn update_centroids(&self, data: &[Vec<f32>], assignments: &[usize]) -> Vec<Vec<f32>> {
        let dimensions = data[0].len();
        let mut new_centroids = vec![vec![0.0; dimensions]; self.num_centroids];
        let mut counts = vec![0; self.num_centroids];
        
        // Accumulate points assigned to each centroid
        for (point, &cluster_idx) in data.iter().zip(assignments.iter()) {
            for (dim, &value) in point.iter().enumerate() {
                new_centroids[cluster_idx][dim] += value;
            }
            counts[cluster_idx] += 1;
        }
        
        // Calculate centroids as cluster means
        for (centroid, count) in new_centroids.iter_mut().zip(counts.iter()) {
            if *count > 0 {
                for value in centroid.iter_mut() {
                    *value /= *count as f32;
                }
            }
        }
        
        new_centroids
    }

    /// Calculate average centroid movement between iterations
    /// 
    /// Measures how much centroids moved by computing the average
    /// Euclidean distance between old and new centroid positions.
    /// This is our primary convergence metric.
    /// 
    /// # Arguments
    /// * `old_centroids` - Previous centroid positions
    /// * `new_centroids` - Current centroid positions
    /// 
    /// # Returns
    /// * Average distance centroids moved
    fn calculate_centroid_shift(&self, old_centroids: &[Vec<f32>], new_centroids: &[Vec<f32>]) -> f32 {
        if old_centroids.is_empty() {
            return 0.0;
        }
        
        old_centroids.iter()
            .zip(new_centroids.iter())
            .map(|(old, new)| euclidean_distance(old, new))
            .sum::<f32>() / old_centroids.len() as f32
    }

    /// Calculate total distortion (sum of squared distances)
    /// 
    /// Measures clustering quality by summing squared distances from
    /// each point to its assigned centroid. Lower distortion indicates
    /// tighter, more cohesive clusters.
    /// 
    /// # Arguments
    /// * `data` - All data points
    /// * `centroids` - Current centroid positions
    /// * `assignments` - Which centroid each point is assigned to
    /// 
    /// # Returns
    /// * Total sum of squared distances
    fn calculate_total_distortion(&self, data: &[Vec<f32>], centroids: &[Vec<f32>], assignments: &[usize]) -> f32 {
        data.iter()
            .zip(assignments.iter())
            .map(|(point, &cluster_idx)| {
                let distance = euclidean_distance(point, &centroids[cluster_idx]);
                distance * distance
            })
            .sum()
    }

    /// Calculate stability score based on cluster population balance
    /// 
    /// Uses entropy to measure how evenly points are distributed across
    /// clusters. Score of 1.0 means perfectly balanced, 0.0 means all
    /// points in one cluster.
    /// 
    /// # Arguments
    /// * `assignments` - Which centroid each point is assigned to
    /// 
    /// # Returns
    /// * Stability score between 0.0 and 1.0
    fn calculate_stability_score(&self, assignments: &[usize]) -> f32 {
        let cluster_counts = self.count_cluster_populations(assignments);
        let total_points = assignments.len() as f32;
        
        if total_points == 0.0 {
            return 0.0;
        }
        
        // Calculate entropy of cluster distribution (higher = more balanced)
        let entropy: f32 = cluster_counts.iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f32 / total_points;
                -p * p.ln()
            })
            .sum();
        
        // Normalize by maximum possible entropy
        let max_entropy = (self.num_centroids as f32).ln();
        if max_entropy > 0.0 { entropy / max_entropy } else { 0.0 }
    }

    /// Count how many points are assigned to each cluster
    /// 
    /// # Arguments
    /// * `assignments` - Which centroid each point is assigned to
    /// 
    /// # Returns
    /// * Vector where counts[i] is number of points in cluster i
    fn count_cluster_populations(&self, assignments: &[usize]) -> Vec<usize> {
        let mut counts = vec![0; self.num_centroids];
        for &assignment in assignments {
            if assignment < counts.len() {
                counts[assignment] += 1;
            }
        }
        counts
    }

    /// Print comprehensive summary of the training process
    /// 
    /// Displays final metrics, cluster populations, and convergence
    /// trajectory to help understand the clustering behavior.
    fn print_convergence_summary(&self) {
        if self.convergence_history.is_empty() { 
            return; 
        }
        
        println!("\n=== Convergence Summary ===");
        
        let final_metrics = self.convergence_history.last().unwrap();
        println!("Final state:");
        println!("  Iterations: {}", final_metrics.iteration);
        println!("  Centroid shift: {:.8}", final_metrics.centroid_shift);
        println!("  Total distortion: {:.2}", final_metrics.total_distortion);
        println!("  Stability score: {:.3}", final_metrics.stability_score);
        
        println!("\nCluster populations:");
        let total_points: usize = final_metrics.cluster_populations.iter().sum();
        for (idx, &population) in final_metrics.cluster_populations.iter().enumerate() {
            if total_points > 0 {
                let percentage = population as f32 / total_points as f32 * 100.0;
                println!("  Cluster {}: {} points ({:.1}%)", idx, population, percentage);
            }
        }
        
        // Show convergence trajectory
        println!("\nConvergence trajectory:");
        let trajectory_length = self.convergence_history.len();
        
        if trajectory_length <= 6 {
            // Show all iterations if short
            for metrics in &self.convergence_history {
                println!("  Iter {}: shift={:.6}, distortion={:.1}", 
                    metrics.iteration, metrics.centroid_shift, metrics.total_distortion);
            }
        } else {
            // Show first few, ellipsis, then last few
            for metrics in self.convergence_history.iter().take(3) {
                println!("  Iter {}: shift={:.6}, distortion={:.1}", 
                    metrics.iteration, metrics.centroid_shift, metrics.total_distortion);
            }
            
            println!("  ...");
            
            for metrics in self.convergence_history.iter().rev().take(3).rev() {
                println!("  Iter {}: shift={:.6}, distortion={:.1}", 
                    metrics.iteration, metrics.centroid_shift, metrics.total_distortion);
            }
        }
    }
}