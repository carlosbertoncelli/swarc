# SWARC Performance Tests

This directory contains performance benchmarks and visualization tools for the SWARC (Small World Approximate Recall Crate) implementation.

## Overview

The performance tests measure SWARC's performance with high-dimensional embeddings (3072 dimensions) across different dataset sizes, from thousands to millions of vectors.

## Test Configuration

- **Embedding Dimension**: 3072 (typical for modern language models)
- **HNSW Parameters**: 
  - `m = 16` (maximum connections per node)
  - `ef_construction = 200` (construction parameter)
- **Test Sizes**: 1K, 10K, 100K, 500K, 1M, 2M, 5M embeddings
- **Search Configuration**: k=10 nearest neighbors

## Running the Benchmarks

### 1. Run the Benchmark

```bash
cargo run --bin benchmark
```

This will:
- Generate random 3072-dimensional embeddings
- Insert them into the HNSW index
- Measure insertion time and throughput
- Perform search operations
- Estimate memory usage
- Save results to `benchmark_results.csv`

### 2. Generate Performance Plots

```bash
cargo run --bin plot_results
```

This will create the following plots:
- `insertion_time.png` - Insertion time vs dataset size
- `search_time.png` - Search time vs dataset size  
- `memory_usage.png` - Memory usage vs dataset size
- `throughput.png` - Insertion throughput vs dataset size

## Performance Results

The benchmarks measure several key metrics:

### Insertion Performance
- **Time Complexity**: O(log N) per insertion
- **Throughput**: Insertions per second
- **Scalability**: Performance across different dataset sizes

### Search Performance
- **Time Complexity**: O(log N) for approximate nearest neighbor search
- **Query Time**: Time to find k nearest neighbors
- **Accuracy**: Approximate recall (not measured in these tests)

### Memory Usage
- **Space Complexity**: O(N × M) where N is number of nodes and M is average connections
- **Memory Efficiency**: Memory usage per embedding
- **Scalability**: Memory growth with dataset size

## Expected Results

Based on the HNSW algorithm characteristics:

- **Insertion Time**: Should scale logarithmically with dataset size
- **Search Time**: Should remain relatively constant (logarithmic scaling)
- **Memory Usage**: Should scale linearly with dataset size
- **Throughput**: May decrease slightly with larger datasets due to increased graph complexity

## Hardware Requirements

For running the full benchmark suite (up to 5M embeddings):

- **RAM**: At least 16GB recommended
- **Storage**: ~2GB for temporary files
- **CPU**: Multi-core processor recommended
- **Time**: Several hours for complete benchmark

## Customization

You can modify the benchmark parameters in `benchmark.rs`:

```rust
let test_sizes = vec![
    1000,      // 1K
    10000,     // 10K
    100000,    // 100K
    500000,    // 500K
    1000000,   // 1M
    2000000,   // 2M
    5000000,   // 5M
];
```

## Output Files

- `benchmark_results.csv` - Raw benchmark data
- `insertion_time.png` - Insertion performance plot
- `search_time.png` - Search performance plot
- `memory_usage.png` - Memory usage plot
- `throughput.png` - Throughput plot

## Notes

- The benchmarks use random embeddings, which may not reflect real-world performance
- Memory usage is estimated and may vary based on system configuration
- Results may vary depending on hardware and system load
- For production use, consider benchmarking with your specific data and use cases
