use swarc::index::HNSWIndex;
use swarc::types::Document;

#[test]
fn test_full_workflow() {
    let mut index = HNSWIndex::new(3, 16, 200);
    
    // Insert multiple documents
    let documents = vec![
        ("doc1", "This is document 1", vec![1.0, 2.0, 3.0]),
        ("doc2", "This is document 2", vec![4.0, 5.0, 6.0]),
        ("doc3", "This is document 3", vec![7.0, 8.0, 9.0]),
        ("doc4", "This is document 4", vec![2.0, 3.0, 4.0]),
        ("doc5", "This is document 5", vec![5.0, 6.0, 7.0]),
    ];
    
    for (id, content, embedding) in documents {
        let doc = Document {
            id: id.to_string(),
            data: content.to_string(),
        };
        index.insert(format!("node_{}", id), embedding, Some(doc)).unwrap();
    }
    
    assert_eq!(index.len(), 5);
    
    // Search for nearest neighbors
    let query = vec![3.0, 4.0, 5.0];
    let results = index.search(&query, 3);
    
    assert_eq!(results.len(), 3);
    
    // Results should be sorted by distance
    for i in 1..results.len() {
        assert!(results[i-1].1 <= results[i].1);
    }
    
    // Remove some documents
    let removed_doc = index.remove("node_doc2").unwrap();
    assert!(removed_doc.is_some());
    assert_eq!(removed_doc.unwrap().data, "This is document 2");
    
    assert_eq!(index.len(), 4);
    
    // Search again after removal
    let results_after = index.search(&query, 3);
    assert_eq!(results_after.len(), 3);
    
    // The removed document should not be in results
    for (id, _distance, _doc) in results_after {
        assert_ne!(id, "node_doc2");
    }
    
    // Rebalance the index
    index.rebalance().unwrap();
    
    // Index should still be functional after rebalancing
    assert_eq!(index.len(), 4);
    let final_results = index.search(&query, 2);
    assert_eq!(final_results.len(), 2);
}

#[test]
fn test_large_dataset_operations() {
    let mut index = HNSWIndex::new(10, 16, 200);
    
    // Insert 100 documents
    for i in 0..100 {
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("Content for document {}", i),
        };
        let embedding: Vec<f32> = (0..10).map(|j| (i + j) as f32).collect();
        index.insert(format!("node_{}", i), embedding, Some(doc)).unwrap();
    }
    
    assert_eq!(index.len(), 100);
    
    // Search for nearest neighbors
    let query: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let results = index.search(&query, 10);
    
    assert_eq!(results.len(), 10);
    
    // Remove multiple documents
    let ids_to_remove: Vec<String> = (0..20).map(|i| format!("node_{}", i)).collect();
    let ids_refs: Vec<&str> = ids_to_remove.iter().map(|s| s.as_str()).collect();
    let removed_docs = index.remove_multiple(&ids_refs).unwrap();
    
    assert_eq!(removed_docs.len(), 20);
    assert_eq!(index.len(), 80);
    
    // Search again
    let results_after = index.search(&query, 10);
    assert_eq!(results_after.len(), 10);
    
    // Clear the index
    index.clear();
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
}

#[test]
fn test_different_data_types() {
    // Test with String data
    let mut index_string = HNSWIndex::new(3, 16, 200);
    let doc_string = Document {
        id: "string_doc".to_string(),
        data: "String content".to_string(),
    };
    index_string.insert("node1".to_string(), vec![1.0, 2.0, 3.0], Some(doc_string)).unwrap();
    
    // Test with numeric data
    let mut index_number = HNSWIndex::new(3, 16, 200);
    let doc_number = Document {
        id: "number_doc".to_string(),
        data: 42,
    };
    index_number.insert("node1".to_string(), vec![1.0, 2.0, 3.0], Some(doc_number)).unwrap();
    
    // Test with vector data
    let mut index_vector = HNSWIndex::new(3, 16, 200);
    let doc_vector = Document {
        id: "vector_doc".to_string(),
        data: vec![1, 2, 3],
    };
    index_vector.insert("node1".to_string(), vec![1.0, 2.0, 3.0], Some(doc_vector)).unwrap();
    
    // All should work for search
    let query = vec![1.0, 2.0, 3.0];
    
    let results_string = index_string.search(&query, 1);
    let results_number = index_number.search(&query, 1);
    let results_vector = index_vector.search(&query, 1);
    
    assert_eq!(results_string.len(), 1);
    assert_eq!(results_number.len(), 1);
    assert_eq!(results_vector.len(), 1);
    
    // All should work for removal
    let removed_string = index_string.remove("node1").unwrap();
    let removed_number = index_number.remove("node1").unwrap();
    let removed_vector = index_vector.remove("node1").unwrap();
    
    assert!(removed_string.is_some());
    assert!(removed_number.is_some());
    assert!(removed_vector.is_some());
    
    assert_eq!(removed_string.unwrap().data, "String content");
    assert_eq!(removed_number.unwrap().data, 42);
    assert_eq!(removed_vector.unwrap().data, vec![1, 2, 3]);
}

#[test]
fn test_search_accuracy() {
    let mut index = HNSWIndex::new(2, 16, 200);
    
    // Insert points in a grid pattern
    let points = vec![
        (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),
        (0.0, 1.0), (1.0, 1.0), (2.0, 1.0),
        (0.0, 2.0), (1.0, 2.0), (2.0, 2.0),
    ];
    
    for (i, (x, y)) in points.iter().enumerate() {
        let doc = Document {
            id: format!("point_{}", i),
            data: format!("Point at ({}, {})", x, y),
        };
        let embedding = vec![*x, *y];
        index.insert(format!("node_{}", i), embedding, Some(doc)).unwrap();
    }
    
    // Search for point closest to (1.5, 1.5)
    let query = vec![1.5, 1.5];
    let results = index.search(&query, 3);
    
    assert_eq!(results.len(), 3);
    
    // The closest point should be (1.0, 1.0) - node_4
    assert_eq!(results[0].0, "node_4");
    
    // Calculate expected distances
    let expected_distances = vec![
        ((1.0, 1.0), 0.707), // sqrt(0.5^2 + 0.5^2)
        ((1.0, 2.0), 0.707), // sqrt(0.5^2 + 0.5^2)
        ((2.0, 1.0), 0.707), // sqrt(0.5^2 + 0.5^2)
    ];
    
    // Check that distances are approximately correct
    for (i, (_, expected_dist)) in expected_distances.iter().enumerate() {
        assert!((results[i].1 - expected_dist).abs() < 0.1);
    }
}

#[test]
fn test_index_persistence_simulation() {
    let mut index = HNSWIndex::new(3, 16, 200);
    
    // Insert documents
    for i in 0..20 {
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("Content {}", i),
        };
        let embedding = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
        index.insert(format!("node_{}", i), embedding, Some(doc)).unwrap();
    }
    
    // Simulate saving state
    let all_ids = index.get_all_ids();
    let node_count = index.len();
    
    // Simulate loading state
    let mut new_index = HNSWIndex::new(3, 16, 200);
    
    // Reinsert all documents
    for i in 0..20 {
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("Content {}", i),
        };
        let embedding = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
        new_index.insert(format!("node_{}", i), embedding, Some(doc)).unwrap();
    }
    
    // Verify state
    assert_eq!(new_index.len(), node_count);
    assert_eq!(new_index.get_all_ids().len(), all_ids.len());
    
    // Test search consistency
    let query = vec![10.0, 11.0, 12.0];
    let results = new_index.search(&query, 5);
    assert_eq!(results.len(), 5);
}

#[test]
fn test_error_handling() {
    let mut index = HNSWIndex::new(3, 16, 200);
    
    // Test duplicate insertion
    let doc = Document {
        id: "doc1".to_string(),
        data: "content1".to_string(),
    };
    let embedding = vec![1.0, 2.0, 3.0];
    
    index.insert("node1".to_string(), embedding.clone(), Some(doc.clone())).unwrap();
    let result = index.insert("node1".to_string(), embedding, Some(doc));
    assert!(result.is_err());
    
    // Test removal of nonexistent node
    let result = index.remove("nonexistent");
    assert!(result.is_err());
    
    // Test removal of multiple nodes with nonexistent ones
    let result = index.remove_multiple(&["node1", "nonexistent"]);
    assert!(result.is_err());
    
    // Index should still be functional
    assert_eq!(index.len(), 1);
    assert!(index.contains("node1"));
}

#[test]
fn test_performance_characteristics() {
    let mut index = HNSWIndex::new(5, 16, 200);
    
    // Insert documents and measure time
    let start = std::time::Instant::now();
    
    for i in 0..1000 {
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("Content {}", i),
        };
        let embedding: Vec<f32> = (0..5).map(|j| (i + j) as f32).collect();
        index.insert(format!("node_{}", i), embedding, Some(doc)).unwrap();
    }
    
    let insert_time = start.elapsed();
    println!("Inserted 1000 documents in {:?}", insert_time);
    
    // Search and measure time
    let start = std::time::Instant::now();
    
    for _ in 0..100 {
        let query: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let _results = index.search(&query, 10);
    }
    
    let search_time = start.elapsed();
    println!("Performed 100 searches in {:?}", search_time);
    
    // Verify results
    assert_eq!(index.len(), 1000);
    
    let query: Vec<f32> = (0..5).map(|i| i as f32).collect();
    let results = index.search(&query, 10);
    assert_eq!(results.len(), 10);
    
    // Results should be sorted by distance
    for i in 1..results.len() {
        assert!(results[i-1].1 <= results[i].1);
    }
}
