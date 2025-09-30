use swarc::{HNSWIndex, Document};
use rand::Rng;

fn main() {
    // Example usage
    let mut index = HNSWIndex::new(128, 16, 200);
    
    // Create some sample documents
    let doc1 = Document {
        id: "doc1".to_string(),
        data: "This is document 1".to_string(),
    };
    
    let doc2 = Document {
        id: "doc2".to_string(),
        data: "This is document 2".to_string(),
    };
    
    // Generate some random embeddings
    let mut rng = rand::thread_rng();
    let embedding1: Vec<f32> = (0..128).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let embedding2: Vec<f32> = (0..128).map(|_| rng.gen_range(-1.0..1.0)).collect();
    
    // Insert documents into the index
    index.insert("node1".to_string(), embedding1.clone(), Some(doc1)).unwrap();
    index.insert("node2".to_string(), embedding2.clone(), Some(doc2)).unwrap();
    
    println!("Index created with {} nodes", index.len());
    
    // Search for nearest neighbors
    let query: Vec<f32> = (0..128).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let results = index.search(&query, 2);
    
    println!("Search results:");
    for (id, distance, document) in results {
        println!("  ID: {}, Distance: {:.4}, Document: {:?}", id, distance, document);
    }
    
    // Remove a node
    if let Ok(removed_doc) = index.remove("node1") {
        println!("Removed document: {:?}", removed_doc);
    }
    
    println!("Index now has {} nodes", index.len());
}
