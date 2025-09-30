use swarc::{HNSWIndex, Document};
use rand::Rng;
use std::time::{Duration, Instant};
use std::fs::File;
use std::io::Write;

#[derive(Debug, Clone)]
struct BenchmarkResult {
    num_embeddings: usize,
    dimension: usize,
    insertion_time: Duration,
    parallel_insertion_time: Duration,
    search_time: Duration,
    memory_usage: usize,
}

fn generate_random_embedding(dimension: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dimension)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect()
}

fn benchmark_insertion(
    index: &mut HNSWIndex<String>,
    embeddings: &[Vec<f32>],
    batch_size: usize,
) -> Duration {
    let start = Instant::now();
    
    for (i, embedding) in embeddings.iter().enumerate() {
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("Content for document {}", i),
        };
        
        index
            .insert(format!("node_{}", i), embedding.clone(), Some(doc))
            .expect("Failed to insert embedding");
        
        // Print progress for large datasets
        if (i + 1) % batch_size == 0 {
            println!("Inserted {} embeddings", i + 1);
        }
    }
    
    start.elapsed()
}

fn benchmark_parallel_insertion(
    index: &mut HNSWIndex<String>,
    embeddings: &[Vec<f32>],
    _batch_size: usize,
) -> Duration {
    let start = Instant::now();
    
    // Prepare all items for parallel insertion
    let items: Vec<(String, Vec<f32>, Option<Document<String>>)> = embeddings
        .iter()
        .enumerate()
        .map(|(i, embedding)| {
            let doc = Document {
                id: format!("doc_{}", i),
                data: format!("Content for document {}", i),
            };
            (format!("node_{}", i), embedding.clone(), Some(doc))
        })
        .collect();
    
    // Insert all items in parallel
    let result = index.insert_parallel(items).expect("Failed to insert embeddings in parallel");
    
    // Check for any errors
    for (i, result) in result.iter().enumerate() {
        if let Err(e) = result {
            panic!("Failed to insert embedding {}: {}", i, e);
        }
    }
    
    // Print progress
    println!("Inserted {} embeddings in parallel", embeddings.len());
    
    start.elapsed()
}

fn benchmark_search(
    index: &HNSWIndex<String>,
    query_embeddings: &[Vec<f32>],
    k: usize,
) -> Duration {
    let start = Instant::now();
    
    for query in query_embeddings {
        let _results = index.search(query, k);
    }
    
    start.elapsed()
}

fn estimate_memory_usage(index: &HNSWIndex<String>) -> usize {
    // Rough estimation of memory usage
    let num_nodes = index.len();
    let dimension = 3072; // Fixed dimension for this benchmark
    let avg_connections = 16; // Average connections per node
    
    // Memory for embeddings + connections + overhead
    num_nodes * (dimension * 4 + avg_connections * 4 + 100) // 4 bytes per f32, 4 bytes per usize, 100 bytes overhead
}

fn run_benchmark(
    num_embeddings: usize,
    dimension: usize,
    batch_size: usize,
) -> BenchmarkResult {
    println!("Starting benchmark with {} embeddings of dimension {}", num_embeddings, dimension);
    
    // Generate embeddings
    println!("Generating {} random embeddings...", num_embeddings);
    let embeddings: Vec<Vec<f32>> = (0..num_embeddings)
        .map(|_| generate_random_embedding(dimension))
        .collect();
    
    // Generate query embeddings (10% of total)
    let num_queries = (num_embeddings / 10).max(100);
    let query_embeddings: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| generate_random_embedding(dimension))
        .collect();
    
    // Benchmark sequential insertion
    println!("Starting sequential insertion benchmark...");
    let mut index_seq = HNSWIndex::new(dimension, 16, 200);
    let insertion_time = benchmark_insertion(&mut index_seq, &embeddings, batch_size);
    
    // Benchmark parallel insertion
    println!("Starting parallel insertion benchmark...");
    let mut index_parallel = HNSWIndex::new(dimension, 16, 200);
    let parallel_insertion_time = benchmark_parallel_insertion(&mut index_parallel, &embeddings, batch_size);
    
    // Benchmark search (using parallel index)
    println!("Starting search benchmark...");
    let search_time = benchmark_search(&index_parallel, &query_embeddings, 10);
    
    // Estimate memory usage
    let memory_usage = estimate_memory_usage(&index_parallel);
    
    BenchmarkResult {
        num_embeddings,
        dimension,
        insertion_time,
        parallel_insertion_time,
        search_time,
        memory_usage,
    }
}

fn save_results_to_csv(results: &[BenchmarkResult], filename: &str) {
    let mut file = File::create(filename).expect("Failed to create CSV file");
    
    // Write header
    writeln!(file, "num_embeddings,dimension,insertion_time_ms,parallel_insertion_time_ms,search_time_ms,memory_usage_mb").unwrap();
    
    // Write data
    for result in results {
        writeln!(
            file,
            "{},{},{},{},{},{}",
            result.num_embeddings,
            result.dimension,
            result.insertion_time.as_millis(),
            result.parallel_insertion_time.as_millis(),
            result.search_time.as_millis(),
            result.memory_usage / (1024 * 1024) // Convert to MB
        ).unwrap();
    }
    
    println!("Results saved to {}", filename);
}

fn main() {
    let dimension = 3072;
    let batch_size = 1000; // Print progress every 1k insertions
    
    // Test different dataset sizes
    let test_sizes = vec![
        1000,      // 1K
        10000,     // 10K
        50000,     // 50K
    ];
    
    let mut results = Vec::new();
    
    for &size in &test_sizes {
        println!("\n=== Benchmarking {} embeddings ===", size);
        
        let result = run_benchmark(size, dimension, batch_size);
        
        // Print summary
        println!("\nBenchmark Summary:");
        println!("  Embeddings: {}", size);
        println!("  Sequential insertion time: {:.2}s", result.insertion_time.as_secs_f64());
        println!("  Parallel insertion time: {:.2}s", result.parallel_insertion_time.as_secs_f64());
        println!("  Speedup: {:.2}x", result.insertion_time.as_secs_f64() / result.parallel_insertion_time.as_secs_f64());
        println!("  Search time: {:.2}ms", result.search_time.as_millis());
        println!("  Memory usage: {:.2}MB", result.memory_usage as f64 / (1024.0 * 1024.0));
        println!("  Sequential insertions/sec: {:.0}", size as f64 / result.insertion_time.as_secs_f64());
        println!("  Parallel insertions/sec: {:.0}", size as f64 / result.parallel_insertion_time.as_secs_f64());
        
        results.push(result);
    }
    
    // Save results to CSV
    save_results_to_csv(&results, "performance_tests/benchmark_results.csv");
    
    println!("\n=== All benchmarks completed ===");
    println!("Results saved to performance_tests/benchmark_results.csv");
}
