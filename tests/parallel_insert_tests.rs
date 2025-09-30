use swarc::index::HNSWIndex;
use swarc::types::Document;
use rand::Rng;

fn generate_random_embedding(dimension: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dimension)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect()
}

#[test]
fn test_parallel_insert_empty_list() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    let result = index.insert_parallel(vec![]);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);
    assert_eq!(index.len(), 0);
}

#[test]
fn test_parallel_insert_single_item() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    let embedding = vec![1.0, 2.0, 3.0];
    let doc = Document {
        id: "doc1".to_string(),
        data: "test data".to_string(),
    };
    
    let items = vec![("node1".to_string(), embedding, Some(doc))];
    let result = index.insert_parallel(items);
    
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].is_ok());
    assert_eq!(index.len(), 1);
    assert!(index.contains("node1"));
}

#[test]
fn test_parallel_insert_multiple_items() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    let items = vec![
        ("node1".to_string(), vec![1.0, 2.0, 3.0], None),
        ("node2".to_string(), vec![4.0, 5.0, 6.0], None),
        ("node3".to_string(), vec![7.0, 8.0, 9.0], None),
    ];
    
    let result = index.insert_parallel(items);
    
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), 3);
    
    for result in &results {
        assert!(result.is_ok());
    }
    
    assert_eq!(index.len(), 3);
    assert!(index.contains("node1"));
    assert!(index.contains("node2"));
    assert!(index.contains("node3"));
}

#[test]
fn test_parallel_insert_with_documents() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    let doc1 = Document {
        id: "doc1".to_string(),
        data: "First document".to_string(),
    };
    
    let doc2 = Document {
        id: "doc2".to_string(),
        data: "Second document".to_string(),
    };
    
    let items = vec![
        ("node1".to_string(), vec![1.0, 2.0, 3.0], Some(doc1)),
        ("node2".to_string(), vec![4.0, 5.0, 6.0], Some(doc2)),
    ];
    
    let result = index.insert_parallel(items);
    
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), 2);
    
    for result in &results {
        assert!(result.is_ok());
    }
    
    assert_eq!(index.len(), 2);
    
    // Verify documents are stored correctly
    let node1 = index.get_node("node1").unwrap();
    assert!(node1.document.is_some());
    assert_eq!(node1.document.as_ref().unwrap().id, "doc1");
    
    let node2 = index.get_node("node2").unwrap();
    assert!(node2.document.is_some());
    assert_eq!(node2.document.as_ref().unwrap().id, "doc2");
}

#[test]
fn test_parallel_insert_duplicate_ids() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    let items = vec![
        ("node1".to_string(), vec![1.0, 2.0, 3.0], None),
        ("node1".to_string(), vec![4.0, 5.0, 6.0], None), // Duplicate ID
    ];
    
    let result = index.insert_parallel(items);
    
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Duplicate ID found"));
    assert_eq!(index.len(), 0);
}

#[test]
fn test_parallel_insert_existing_id() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert first node
    index.insert("node1".to_string(), vec![1.0, 2.0, 3.0], None).unwrap();
    
    let items = vec![
        ("node1".to_string(), vec![4.0, 5.0, 6.0], None), // Existing ID
        ("node2".to_string(), vec![7.0, 8.0, 9.0], None),
    ];
    
    let result = index.insert_parallel(items);
    
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("already exists in index"));
    assert_eq!(index.len(), 1); // Only the original node should exist
}

#[test]
fn test_parallel_insert_large_dataset() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(128, 16, 200);
    
    let num_items = 1000;
    let mut items = Vec::new();
    
    for i in 0..num_items {
        let embedding = generate_random_embedding(128);
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("Content for document {}", i),
        };
        
        items.push((format!("node_{}", i), embedding, Some(doc)));
    }
    
    let result = index.insert_parallel(items);
    
    assert!(result.is_ok());
    let results = result.unwrap();
    assert_eq!(results.len(), num_items);
    
    for result in &results {
        assert!(result.is_ok());
    }
    
    assert_eq!(index.len(), num_items);
    
    // Verify some random nodes exist
    assert!(index.contains("node_0"));
    assert!(index.contains("node_500"));
    assert!(index.contains("node_999"));
}

#[test]
fn test_parallel_vs_sequential_insertion() {
    let dimension = 64;
    let num_items = 100;
    
    // Test parallel insertion
    let mut index_parallel: HNSWIndex<String> = HNSWIndex::new(dimension, 16, 200);
    let mut items_parallel = Vec::new();
    
    for i in 0..num_items {
        let embedding = generate_random_embedding(dimension);
        items_parallel.push((format!("node_{}", i), embedding, None));
    }
    
    let parallel_result = index_parallel.insert_parallel(items_parallel);
    assert!(parallel_result.is_ok());
    
    // Test sequential insertion
    let mut index_sequential: HNSWIndex<String> = HNSWIndex::new(dimension, 16, 200);
    let mut items_sequential = Vec::new();
    
    for i in 0..num_items {
        let embedding = generate_random_embedding(dimension);
        items_sequential.push((format!("node_{}", i), embedding, None));
    }
    
    let sequential_result = index_sequential.insert_multiple(items_sequential);
    assert!(sequential_result.is_ok());
    
    // Both should have the same number of nodes
    assert_eq!(index_parallel.len(), index_sequential.len());
    assert_eq!(index_parallel.len(), num_items);
    
    // Both should be searchable
    let query = generate_random_embedding(dimension);
    let parallel_search = index_parallel.search(&query, 5);
    let sequential_search = index_sequential.search(&query, 5);
    
    assert_eq!(parallel_search.len(), sequential_search.len());
    assert_eq!(parallel_search.len(), 5);
}

#[test]
fn test_parallel_insertion_search_accuracy() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert known vectors
    let items = vec![
        ("close1".to_string(), vec![1.0, 1.0, 1.0], None),
        ("close2".to_string(), vec![1.1, 1.1, 1.1], None),
        ("far1".to_string(), vec![10.0, 10.0, 10.0], None),
        ("far2".to_string(), vec![11.0, 11.0, 11.0], None),
    ];
    
    let result = index.insert_parallel(items);
    assert!(result.is_ok());
    
    // Search for a vector close to the first two
    let query = vec![1.05, 1.05, 1.05];
    let results = index.search(&query, 2);
    
    assert_eq!(results.len(), 2);
    
    // The closest results should be the "close" nodes
    let result_ids: Vec<&str> = results.iter().map(|(id, _, _)| id.as_str()).collect();
    assert!(result_ids.contains(&"close1") || result_ids.contains(&"close2"));
}

#[test]
fn test_parallel_insertion_with_different_dimensions() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(5, 16, 200);
    
    let items = vec![
        ("node1".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0], None),
        ("node2".to_string(), vec![6.0, 7.0, 8.0, 9.0, 10.0], None),
    ];
    
    let result = index.insert_parallel(items);
    assert!(result.is_ok());
    
    assert_eq!(index.len(), 2);
    
    // Test search with the same dimension
    let query = vec![1.5, 2.5, 3.5, 4.5, 5.5];
    let results = index.search(&query, 1);
    
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "node1"); // Should be closest to node1
}

#[test]
fn test_parallel_insertion_error_handling() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Test with invalid embedding dimensions (should not panic, but may have issues)
    let items = vec![
        ("node1".to_string(), vec![1.0, 2.0], None), // Wrong dimension
        ("node2".to_string(), vec![3.0, 4.0, 5.0], None), // Correct dimension
    ];
    
    // This should still work, but the search might not be optimal
    let result = index.insert_parallel(items);
    assert!(result.is_ok());
    
    assert_eq!(index.len(), 2);
}
