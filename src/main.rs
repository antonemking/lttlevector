// src/main.rs
//! LittleVector: Educational Vector Database with Production Capabilities
//!
//! This demonstrates both the educational foundations and realistic scale
//! capabilities of LittleVector in a clean, user-facing interface.

use littlevector::{LittleVector, presets, demo};
use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_usage();
        return;
    }
    
    match args[1].as_str() {
        "demo" => run_demo_mode(&args[2..]),
        "educational" => run_educational_mode(),
        "performance" => run_performance_mode(), 
        "compare" => run_comparison_mode(),
        "documents" => run_document_mode(),
        "--help" | "-h" => print_help(),
        "--version" | "-v" => print_version(),
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage();
            process::exit(1);
        }
    }
}

fn run_demo_mode(args: &[String]) {
    println!("🚀 LittleVector Demo Mode");
    println!("=========================");
    
    if args.is_empty() {
        // Run default demo sequence
        run_complete_demo();
    } else {
        match args[0].as_str() {
            "quick" => run_quick_demo(),
            "full" => run_complete_demo(),
            "interactive" => run_interactive_demo(),
            _ => {
                eprintln!("Unknown demo type: {}", args[0]);
                eprintln!("Available: quick, full, interactive");
            }
        }
    }
}

fn run_educational_mode() {
    println!("📚 LittleVector Educational Mode");
    println!("=================================");
    println!("This mode demonstrates core concepts with small, interpretable data.\n");
    
    match demo::run_educational_demo() {
        Ok(()) => {
            println!("\n✅ Educational demo completed successfully!");
            println!("\n📖 What you learned:");
            println!("  • Product Quantization compresses vectors 100-400x");
            println!("  • K-means clustering finds natural centroids"); 
            println!("  • Asymmetric search enables fast similarity queries");
            println!("  • Mathematical transparency shows exactly how it works");
        }
        Err(e) => {
            eprintln!("❌ Educational demo failed: {}", e);
            process::exit(1);
        }
    }
}

fn run_performance_mode() {
    println!("🎯 LittleVector Performance Mode");
    println!("=================================");
    println!("Testing with realistic 768-dimensional vectors and thousands of documents.\n");
    
    match demo::run_performance_demo() {
        Ok(()) => {
            println!("\n✅ Performance demo completed successfully!");
            println!("\n📊 Key insights:");
            println!("  • Sub-millisecond search across thousands of documents");
            println!("  • 300-400x compression ratio on realistic embeddings");
            println!("  • Memory usage scales linearly with dataset size");
            println!("  • Training converges quickly with k-means++ initialization");
        }
        Err(e) => {
            eprintln!("❌ Performance demo failed: {}", e);
            process::exit(1);
        }
    }
}

fn run_comparison_mode() {
    println!("⚖️  LittleVector Configuration Comparison");
    println!("=========================================");
    println!("Comparing different configuration presets on the same dataset.\n");
    
    match demo::run_configuration_comparison() {
        Ok(()) => {
            println!("\n✅ Comparison completed successfully!");
            println!("\n🔧 Configuration guide:");
            println!("  • Educational: Best for learning and small datasets");
            println!("  • Production: Balanced performance and accuracy");
            println!("  • High Compression: Maximum space efficiency");
        }
        Err(e) => {
            eprintln!("❌ Comparison failed: {}", e);
            process::exit(1);
        }
    }
}

fn run_document_mode() {
    println!("📄 LittleVector Document Search");
    println!("===============================");
    println!("Demonstrating semantic search over document embeddings.\n");
    
    match demo::run_document_search_demo() {
        Ok(()) => {
            println!("\n✅ Document search demo completed!");
            println!("\n📝 Applications:");
            println!("  • Legal document retrieval");
            println!("  • Policy and contract search");
            println!("  • Knowledge base queries");
            println!("  • Content recommendation");
        }
        Err(e) => {
            eprintln!("❌ Document demo failed: {}", e);
            process::exit(1);
        }
    }
}

fn run_quick_demo() {
    println!("⚡ Quick Demo: LittleVector in 30 seconds\n");
    
    // Generate small dataset
    let docs = demo::generate_sample_data(100, Some(42));
    println!("📊 Generated {} test documents (768D)", docs.len());
    
    // Create database with production config
    let mut db = LittleVector::with_config(presets::production());
    
    // Train
    let start = std::time::Instant::now();
    match db.train(&docs[..50]) {
        Ok(result) => println!("🧠 Trained in {:.1}ms", result.training_time_ms),
        Err(e) => {
            eprintln!("Training failed: {}", e);
            return;
        }
    }
    
    // Add documents
    match db.add_documents(&docs[50..]) {
        Ok(result) => println!("🗜️  Compressed {} docs ({:.1}x ratio)", 
                             result.documents_added, result.compression_ratio),
        Err(e) => {
            eprintln!("Compression failed: {}", e);
            return;
        }
    }
    
    // Search
    match db.search(&docs[0].embedding, 5) {
        Ok(results) => {
            println!("🔍 Search completed in {:.3}ms", results.search_time_ms);
            println!("   Top result: {} (distance: {:.3})", 
                     results.documents[0].id, results.documents[0].distance);
        }
        Err(e) => {
            eprintln!("Search failed: {}", e);
            return;
        }
    }
    
    let total_time = start.elapsed();
    println!("\n✅ Complete workflow in {:.1}ms", total_time.as_secs_f32() * 1000.0);
}

fn run_complete_demo() {
    println!("🎬 Complete Demo: Full LittleVector Capabilities\n");
    
    println!("Part 1: Educational Foundation");
    println!("─────────────────────────────");
    if let Err(e) = demo::run_educational_demo() {
        eprintln!("Educational demo failed: {}", e);
        return;
    }
    
    println!("\n\nPart 2: Performance at Scale");
    println!("────────────────────────────");
    if let Err(e) = demo::run_performance_demo() {
        eprintln!("Performance demo failed: {}", e);
        return;
    }
    
    println!("\n\nPart 3: Configuration Options");
    println!("─────────────────────────────");
    if let Err(e) = demo::run_configuration_comparison() {
        eprintln!("Configuration comparison failed: {}", e);
        return;
    }
    
    println!("\n\n🎉 Complete demo finished!");
    println!("You've seen LittleVector from educational concepts to production capabilities.");
}

fn run_interactive_demo() {
    println!("🎮 Interactive Demo Mode");
    println!("========================");
    println!("This would be an interactive CLI for exploring LittleVector features.");
    println!("(Interactive mode not implemented in this demo)");
    
    // TODO: Implement interactive CLI
    // - Let users choose dataset size
    // - Configure parameters
    // - Run custom queries
    // - Visualize results
}

fn print_usage() {
    println!("Usage: littlevector <COMMAND> [OPTIONS]");
    println!("");
    println!("Commands:");
    println!("  demo [quick|full|interactive]  Run demonstration modes");
    println!("  educational                    Learn core concepts with small data");
    println!("  performance                    Test realistic scale performance");
    println!("  compare                        Compare configuration presets");
    println!("  documents                      Semantic document search demo");
    println!("  --help, -h                     Show this help message");
    println!("  --version, -v                  Show version information");
    println!("");
    println!("Examples:");
    println!("  littlevector demo quick        # 30-second overview");
    println!("  littlevector educational       # Learn the fundamentals");
    println!("  littlevector performance       # Realistic scale testing");
    println!("  littlevector compare           # Configuration comparison");
}

fn print_help() {
    println!("LittleVector - Educational Vector Database");
    println!("==========================================");
    println!("");
    print_usage();
    println!("");
    println!("About:");
    println!("  LittleVector is a transparent implementation of Product Quantization");
    println!("  and vector similarity search. It scales from educational demos to"); 
    println!("  realistic workloads while maintaining mathematical clarity.");
    println!("");
    println!("Features:");
    println!("  • Product Quantization compression (100-400x)");
    println!("  • K-means clustering for centroid training");
    println!("  • Asymmetric distance computation for fast search");
    println!("  • Configurable for different use cases");
    println!("  • Educational transparency at every step");
    println!("");
    println!("Educational Purpose:");
    println!("  Unlike production vector databases, LittleVector exposes the");
    println!("  mathematical operations so you can understand exactly how");
    println!("  modern vector search systems work under the hood.");
    println!("");
    println!("Repository: https://github.com/your-username/littlevector");
}

fn print_version() {
    println!("LittleVector v{}", env!("CARGO_PKG_VERSION"));
    println!("Educational Vector Database with Production Capabilities");
    println!("Built with Rust for performance and mathematical transparency");
}

// Integration test that can be run as part of main
#[cfg(test)]
mod integration_tests {
    use super::*;
    use littlevector::*;

    #[test]
    fn test_complete_workflow() {
        let docs = demo::generate_sample_data(50, Some(123));
        let mut db = LittleVector::with_config(presets::educational());
        
        // Should complete without errors
        db.train(&docs[..25]).unwrap();
        db.add_documents(&docs[25..]).unwrap();
        let results = db.search(&docs[0].embedding, 5).unwrap();
        
        assert_eq!(results.documents.len(), 5);
        assert!(results.search_time_ms < 100.0); // Reasonable performance
    }
}