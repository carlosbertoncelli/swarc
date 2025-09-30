use swarc::index::HNSWIndex;
use swarc::types::Document;

#[test]
fn test_search_empty_index() {
    let index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    let query = vec![1.0, 2.0, 3.0];
    let results = index.search(&query, 5);
    
    assert!(results.is_empty());
}

#[test]
fn test_search_single_node() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    let doc = Document {
        id: "test_doc".to_string(),
        data: "test content".to_string(),
    };
    
    let embedding = vec![1.0, 2.0, 3.0];
    index.insert("node1".to_string(), embedding, Some(doc)).unwrap();
    
    let query = vec![1.0, 2.0, 3.0];
    let results = index.search(&query, 1);
    
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "node1");
    assert!((results[0].1 - 0.0).abs() < 1e-6); // Distance should be 0
    assert!(results[0].2.is_some());
}

#[test]
fn test_search_multiple_nodes() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert nodes with known distances
    let embeddings = vec![
        vec![0.0, 0.0, 0.0],  // node0
        vec![1.0, 0.0, 0.0],  // node1 - distance 1 from origin
        vec![2.0, 0.0, 0.0],  // node2 - distance 2 from origin
        vec![0.0, 1.0, 0.0],  // node3 - distance 1 from origin
        vec![0.0, 0.0, 1.0],  // node4 - distance 1 from origin
    ];
    
    for (i, embedding) in embeddings.iter().enumerate() {
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("content_{}", i),
        };
        index.insert(format!("node_{}", i), embedding.clone(), Some(doc)).unwrap();
    }
    
    let query = vec![0.0, 0.0, 0.0];
    let results = index.search(&query, 3);
    
    assert_eq!(results.len(), 3);
    
    // Results should be sorted by distance
    for i in 1..results.len() {
        assert!(results[i-1].1 <= results[i].1);
    }
    
    // First result should be node0 (distance 0)
    assert_eq!(results[0].0, "node_0");
    assert!((results[0].1 - 0.0).abs() < 1e-6);
}

#[test]
fn test_search_k_larger_than_nodes() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert only 2 nodes
    for i in 0..2 {
        let embedding = vec![i as f32, 0.0, 0.0];
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("content_{}", i),
        };
        index.insert(format!("node_{}", i), embedding, Some(doc)).unwrap();
    }
    
    let query = vec![0.0, 0.0, 0.0];
    let results = index.search(&query, 5); // Request 5 but only 2 available
    
    assert_eq!(results.len(), 2);
}

#[test]
fn test_search_with_different_query() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert nodes
    for i in 0..5 {
        let embedding = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("content_{}", i),
        };
        index.insert(format!("node_{}", i), embedding, Some(doc)).unwrap();
    }
    
    // Search with a query that's close to one of the nodes
    let query = vec![2.0, 3.0, 4.0]; // Close to node_2
    let results = index.search(&query, 3);
    
    assert_eq!(results.len(), 3);
    
    // Results should be sorted by distance
    for i in 1..results.len() {
        assert!(results[i-1].1 <= results[i].1);
    }
}

#[test]
fn test_search_layer() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert nodes to create a small graph
    for i in 0..5 {
        let embedding = vec![i as f32, 0.0, 0.0];
        index.insert(format!("node_{}", i), embedding, None).unwrap();
    }
    
    let query = vec![0.0, 0.0, 0.0];
    let _entry_points = vec![0, 1, 2];
    
    // Test search_layer (this is a private method, so we test it indirectly)
    // by checking that search works correctly
    let results = index.search(&query, 3);
    
    assert_eq!(results.len(), 3);
    
    // All results should have valid node IDs
    for (id, _distance, _doc) in &results {
        assert!(id.starts_with("node_"));
    }
}

#[test]
fn test_search_with_zero_query() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert nodes with different embeddings
    let embeddings = vec![
        vec![1.0, 1.0, 1.0],
        vec![2.0, 2.0, 2.0],
        vec![3.0, 3.0, 3.0],
    ];
    
    for (i, embedding) in embeddings.iter().enumerate() {
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("content_{}", i),
        };
        index.insert(format!("node_{}", i), embedding.clone(), Some(doc)).unwrap();
    }
    
    let zero_query = vec![0.0, 0.0, 0.0];
    let results = index.search(&zero_query, 2);
    
    assert_eq!(results.len(), 2);
    
    // Results should be sorted by distance from origin
    for i in 1..results.len() {
        assert!(results[i-1].1 <= results[i].1);
    }
}

#[test]
fn test_search_with_negative_query() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert nodes with positive embeddings
    for i in 0..3 {
        let embedding = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("content_{}", i),
        };
        index.insert(format!("node_{}", i), embedding, Some(doc)).unwrap();
    }
    
    let negative_query = vec![-1.0, -2.0, -3.0];
    let results = index.search(&negative_query, 2);
    
    assert_eq!(results.len(), 2);
    
    // All distances should be positive
    for (_id, distance, _doc) in &results {
        assert!(*distance >= 0.0);
    }
}

#[test]
fn test_search_without_documents() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert nodes without documents
    for i in 0..3 {
        let embedding = vec![i as f32, 0.0, 0.0];
        index.insert(format!("node_{}", i), embedding, None).unwrap();
    }
    
    let query = vec![0.0, 0.0, 0.0];
    let results = index.search(&query, 3);
    
    assert_eq!(results.len(), 3);
    
    // All documents should be None
    for (_id, _distance, doc) in &results {
        assert!(doc.is_none());
    }
}

#[test]
fn test_search_consistency() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert nodes
    for i in 0..10 {
        let embedding = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("content_{}", i),
        };
        index.insert(format!("node_{}", i), embedding, Some(doc)).unwrap();
    }
    
    let query = vec![5.0, 6.0, 7.0];
    
    // Run search multiple times - results should be consistent
    let results1 = index.search(&query, 5);
    let results2 = index.search(&query, 5);
    
    assert_eq!(results1.len(), results2.len());
    
    // Results should be the same (deterministic search)
    for i in 0..results1.len() {
        assert_eq!(results1[i].0, results2[i].0);
        assert!((results1[i].1 - results2[i].1).abs() < 1e-6);
    }
}

#[test]
fn test_search_large_k() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert 20 nodes
    for i in 0..20 {
        let embedding = vec![i as f32, 0.0, 0.0];
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("content_{}", i),
        };
        index.insert(format!("node_{}", i), embedding, Some(doc)).unwrap();
    }
    
    let query = vec![10.0, 0.0, 0.0];
    let results = index.search(&query, 15);
    
    assert_eq!(results.len(), 15);
    
    // Results should be sorted by distance
    for i in 1..results.len() {
        assert!(results[i-1].1 <= results[i].1);
    }
}
