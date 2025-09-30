# SWARC - Small World Approximate Recall Crate

A high-performance implementation of the Hierarchical Navigable Small World (HNSW) algorithm in Rust. SWARC provides state-of-the-art approximate nearest neighbor search with excellent performance for high-dimensional vector similarity search.

## Features

- **Fast Approximate Nearest Neighbor Search**: Efficient k-NN search with logarithmic time complexity
- **Multiple Distance Metrics**: Support for Euclidean and Cosine distance calculations
- **Document Linking**: Associate embeddings with external data using a flexible document structure
- **Dynamic Operations**: Insert, remove, and rebalance the index at runtime
- **Modular Architecture**: Clean separation of concerns with dedicated modules for different operations
- **Type Safety**: Generic implementation that works with any data type
- **Serialization Support**: Built-in support for JSON serialization/deserialization

## Algorithm Overview

HNSW constructs a multi-layer graph where:
- Higher layers contain fewer nodes and provide long-range connections
- Lower layers contain more nodes and provide fine-grained search
- Search starts from the top layer and navigates down to find nearest neighbors
- Insertion uses probabilistic level assignment and intelligent neighbor selection

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
swarc = { path = "path/to/swarc" }
```

Or if using from crates.io:

```toml
[dependencies]
swarc = "0.1.0"
```

## Quick Start

```rust
use swarc::{HNSWIndex, Document, DistanceMetric};
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new HNSW index with Euclidean distance (default)
    // Parameters: dimension, max_connections, ef_construction
    let mut index = HNSWIndex::new(128, 16, 200);
    
    // Or create with a specific distance metric
    let mut index_cosine = HNSWIndex::new_with_distance(128, 16, 200, DistanceMetric::Cosine);
    
    // Create a document with external data
    let document = Document {
        id: "doc1".to_string(),
        data: "This is my document content".to_string(),
    };
    
    // Generate a random embedding (in practice, use your embedding model)
    let mut rng = rand::thread_rng();
    let embedding: Vec<f32> = (0..128).map(|_| rng.gen_range(-1.0..1.0)).collect();
    
    // Insert the document into the index
    index.insert("node1".to_string(), embedding, Some(document))?;
    
    // Search for nearest neighbors
    let query: Vec<f32> = (0..128).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let results = index.search(&query, 5); // Find 5 nearest neighbors
    
    for (id, distance, document) in results {
        println!("ID: {}, Distance: {:.4}, Document: {:?}", id, distance, document);
    }
    
    // Remove a document
    let removed_doc = index.remove("node1")?;
    println!("Removed: {:?}", removed_doc);
    
    Ok(())
}
```

## API Reference

### Core Types

#### `Document<T>`
```rust
pub struct Document<T> {
    pub id: String,
    pub data: T,
}
```
A wrapper for external data associated with embeddings.

#### `DistanceMetric`
```rust
pub enum DistanceMetric {
    Euclidean,
    Cosine,
}
```
Distance metrics supported by the HNSW index.

#### `HNSWIndex<T>`
The main index structure that provides all HNSW operations.

### Main Methods

#### `new(dim: usize, m: usize, ef_construction: usize) -> HNSWIndex<T>`
Creates a new HNSW index with Euclidean distance (default).

- `dim`: Dimensionality of the embedding vectors
- `m`: Maximum number of connections per node (except layer 0)
- `ef_construction`: Size of dynamic candidate list during construction

#### `new_with_distance(dim: usize, m: usize, ef_construction: usize, distance_metric: DistanceMetric) -> HNSWIndex<T>`
Creates a new HNSW index with a specific distance metric.

- `dim`: Dimensionality of the embedding vectors
- `m`: Maximum number of connections per node (except layer 0)
- `ef_construction`: Size of dynamic candidate list during construction
- `distance_metric`: The distance metric to use (`DistanceMetric::Euclidean` or `DistanceMetric::Cosine`)

#### `insert(id: String, embedding: Vec<f32>, document: Option<Document<T>>) -> Result<(), String>`
Inserts a new node into the index.

- `id`: Unique identifier for the node
- `embedding`: The vector embedding
- `document`: Optional associated document data

#### `search(query: &[f32], k: usize) -> Vec<(String, f32, Option<&Document<T>>)>`
Searches for k nearest neighbors.

- `query`: The query vector
- `k`: Number of nearest neighbors to return
- Returns: Vector of (node_id, distance, document) tuples

#### `remove(id: &str) -> Result<Option<Document<T>>, String>`
Removes a node from the index.

- `id`: The node identifier to remove
- Returns: The removed document if it existed

#### `rebalance() -> Result<(), String>`
Rebalances the index structure (currently a placeholder for future enhancements).

### Utility Methods

- `len() -> usize`: Get the number of nodes in the index
- `is_empty() -> bool`: Check if the index is empty
- `get_node(id: &str) -> Option<&HNSWNode<T>>`: Get a node by ID
- `get_all_ids() -> Vec<String>`: Get all node IDs
- `contains(id: &str) -> bool`: Check if a node exists
- `clear()`: Remove all nodes from the index
- `remove_multiple(ids: &[&str]) -> Result<Vec<Option<Document<T>>>, String>`: Remove multiple nodes

## Architecture

The implementation is organized into several modules:

- **`types.rs`**: Core data structures (`Document`, `HNSWNode`)
- **`index.rs`**: Main index structure and basic operations
- **`insert.rs`**: Insertion and rebalancing logic
- **`search.rs`**: Search and nearest neighbor algorithms
- **`remove.rs`**: Node removal and cleanup operations

## Performance Characteristics

- **Search Complexity**: O(log N) for approximate nearest neighbor search
- **Insertion Complexity**: O(log N) for adding new nodes
- **Memory Usage**: O(N × M) where N is the number of nodes and M is the average connections per node
- **Distance Metrics**: Supports Euclidean distance (L2 norm) and Cosine distance

### Performance Benchmarks

SWARC has been benchmarked with high-dimensional embeddings (3072 dimensions) across various dataset sizes. See the [performance tests](performance_tests/) directory for detailed benchmarks and visualizations.

**Key Performance Metrics:**
- **Insertion Throughput**: Up to millions of embeddings per hour
- **Search Time**: Sub-millisecond query times for large datasets
- **Memory Efficiency**: Linear scaling with dataset size
- **Scalability**: Tested up to 5 million 3072-dimensional embeddings

![Insertion Performance](performance_tests/insertion_time.png)
*Insertion time scales logarithmically with dataset size*

![Search Performance](performance_tests/search_time.png)
*Search time remains relatively constant across dataset sizes*

![Memory Usage](performance_tests/memory_usage.png)
*Memory usage scales linearly with dataset size*

![Throughput](performance_tests/throughput.png)
*Insertion throughput across different dataset sizes*

## Configuration Parameters

### `m` (Maximum Connections)
- Controls the maximum number of connections per node
- Higher values: Better recall, more memory usage, slower search
- Lower values: Faster search, less memory, potentially lower recall
- Typical range: 8-32

### `ef_construction` (Construction Parameter)
- Size of dynamic candidate list during index construction
- Higher values: Better index quality, slower construction
- Lower values: Faster construction, potentially lower search quality
- Typical range: 100-500

### `max_layers`
- Automatically calculated based on expected dataset size
- Higher layers provide long-range navigation
- Lower layers provide fine-grained search

### Distance Metrics

#### Euclidean Distance (Default)
- Measures straight-line distance in vector space
- Good for general-purpose similarity search
- Sensitive to vector magnitude
- Formula: `sqrt(sum((a_i - b_i)²))`

#### Cosine Distance
- Measures angular similarity between vectors
- Good for text embeddings and normalized vectors
- Magnitude-invariant (only considers direction)
- Formula: `1 - (dot_product / (norm_a * norm_b))`

**When to use each metric:**
- **Euclidean**: When vector magnitude matters (e.g., image embeddings, raw feature vectors)
- **Cosine**: When only direction matters (e.g., text embeddings, normalized vectors)

## Examples

### Basic Usage
```rust
// Euclidean distance (default)
let mut index = HNSWIndex::new(128, 16, 200);
index.insert("node1".to_string(), vec![0.1, 0.2, 0.3], None)?;
let results = index.search(&vec![0.1, 0.2, 0.3], 1);

// Cosine distance
let mut index_cosine = HNSWIndex::new_with_distance(128, 16, 200, DistanceMetric::Cosine);
index_cosine.insert("node1".to_string(), vec![0.1, 0.2, 0.3], None)?;
let results_cosine = index_cosine.search(&vec![0.1, 0.2, 0.3], 1);
```

### Performance Testing
```bash
# Run benchmarks with millions of 3072-dimensional embeddings
cargo run --bin benchmark

# Generate performance plots
cargo run --bin plot_results
```

### With Documents
```rust
let doc = Document {
    id: "article_1".to_string(),
    data: Article { title: "AI Research", content: "..." },
};
index.insert("node1".to_string(), embedding, Some(doc))?;
```

### Distance Metric Comparison
```rust
use swarc::{HNSWIndex, DistanceMetric};

// Create indices with different distance metrics
let mut index_euclidean = HNSWIndex::new_with_distance(128, 16, 200, DistanceMetric::Euclidean);
let mut index_cosine = HNSWIndex::new_with_distance(128, 16, 200, DistanceMetric::Cosine);

// Insert the same data
let embedding1 = vec![1.0, 0.0, 0.0];
let embedding2 = vec![2.0, 0.0, 0.0]; // Same direction, different magnitude
let embedding3 = vec![0.0, 1.0, 0.0]; // Orthogonal direction

index_euclidean.insert("doc1".to_string(), embedding1.clone(), None)?;
index_euclidean.insert("doc2".to_string(), embedding2.clone(), None)?;
index_euclidean.insert("doc3".to_string(), embedding3.clone(), None)?;

index_cosine.insert("doc1".to_string(), embedding1.clone(), None)?;
index_cosine.insert("doc2".to_string(), embedding2.clone(), None)?;
index_cosine.insert("doc3".to_string(), embedding3.clone(), None)?;

// Search with query vector
let query = vec![0.5, 0.0, 0.0];
let results_euclidean = index_euclidean.search(&query, 2);
let results_cosine = index_cosine.search(&query, 2);

// Results will differ based on distance metric:
// Euclidean: doc1 (closest), doc2 (farther due to magnitude)
// Cosine: doc1 and doc2 (equally close due to same direction)
```

### Batch Operations
```rust
// Insert multiple documents
for (i, embedding) in embeddings.iter().enumerate() {
    let doc = Document {
        id: format!("doc_{}", i),
        data: documents[i].clone(),
    };
    index.insert(format!("node_{}", i), embedding.clone(), Some(doc))?;
}

// Remove multiple documents
let ids = ["node_1", "node_2", "node_3"];
let removed = index.remove_multiple(&ids)?;
```

## Limitations and Future Work

### Current Limitations
- Rebalancing is a placeholder implementation
- No persistence/serialization of the index structure
- Limited to Euclidean and Cosine distance metrics

### Planned Enhancements
- Support for additional distance metrics (inner product, Manhattan, etc.)
- Index persistence and loading
- Advanced rebalancing strategies
- Parallel search and insertion
- Memory-mapped storage for large datasets
- Benchmarking and performance optimization tools

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) - Original HNSW paper
- [HNSW: Hierarchical Navigable Small World](https://github.com/nmslib/hnswlib) - Reference C++ implementation
