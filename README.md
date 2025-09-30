# SWARC - Small World Approximate Recall Crate

A high-performance implementation of the Hierarchical Navigable Small World (HNSW) algorithm in Rust. SWARC provides state-of-the-art approximate nearest neighbor search with excellent performance for high-dimensional vector similarity search.

## Features

- **Fast Approximate Nearest Neighbor Search**: Efficient k-NN search with logarithmic time complexity
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

Or if using from crates.io (when published):

```toml
[dependencies]
swarc = "0.1.0"
```

## Quick Start

```rust
use swarc::{HNSWIndex, Document};
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new HNSW index
    // Parameters: dimension, max_connections, ef_construction
    let mut index = HNSWIndex::new(128, 16, 200);
    
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

#### `HNSWIndex<T>`
The main index structure that provides all HNSW operations.

### Main Methods

#### `new(dim: usize, m: usize, ef_construction: usize) -> HNSWIndex<T>`
Creates a new HNSW index.

- `dim`: Dimensionality of the embedding vectors
- `m`: Maximum number of connections per node (except layer 0)
- `ef_construction`: Size of dynamic candidate list during construction

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
- **Distance Metric**: Currently uses Euclidean distance (L2 norm)

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

## Examples

### Basic Usage
```rust
let mut index = HNSWIndex::new(128, 16, 200);
index.insert("node1".to_string(), vec![0.1, 0.2, 0.3], None)?;
let results = index.search(&vec![0.1, 0.2, 0.3], 1);
```

### With Documents
```rust
let doc = Document {
    id: "article_1".to_string(),
    data: Article { title: "AI Research", content: "..." },
};
index.insert("node1".to_string(), embedding, Some(doc))?;
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
- Only supports Euclidean distance (L2 norm)
- Rebalancing is a placeholder implementation
- No persistence/serialization of the index structure
- No support for different distance metrics

### Planned Enhancements
- Support for multiple distance metrics (cosine, inner product, etc.)
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
