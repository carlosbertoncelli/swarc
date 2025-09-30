use swarc::index::HNSWIndex;
use swarc::types::Document;

#[test]
fn test_insert_single_node() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    let doc = Document {
        id: "test_doc".to_string(),
        data: "test content".to_string(),
    };
    
    let embedding = vec![1.0, 2.0, 3.0];
    let result = index.insert("node1".to_string(), embedding, Some(doc));
    
    assert!(result.is_ok());
    assert_eq!(index.len(), 1);
    assert!(!index.is_empty());
    assert_eq!(index.get_all_ids(), vec!["node1"]);
}

#[test]
fn test_insert_multiple_nodes() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    for i in 0..5 {
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("content_{}", i),
        };
        
        let embedding = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
        let result = index.insert(format!("node_{}", i), embedding, Some(doc));
        
        assert!(result.is_ok());
    }
    
    assert_eq!(index.len(), 5);
    assert_eq!(index.get_all_ids().len(), 5);
}

#[test]
fn test_insert_duplicate_id() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    let doc1 = Document {
        id: "doc1".to_string(),
        data: "content1".to_string(),
    };
    
    let doc2 = Document {
        id: "doc2".to_string(),
        data: "content2".to_string(),
    };
    
    let embedding1 = vec![1.0, 2.0, 3.0];
    let embedding2 = vec![4.0, 5.0, 6.0];
    
    // First insertion should succeed
    let result1 = index.insert("node1".to_string(), embedding1, Some(doc1));
    assert!(result1.is_ok());
    
    // Second insertion with same ID should fail
    let result2 = index.insert("node1".to_string(), embedding2, Some(doc2));
    assert!(result2.is_err());
    assert!(result2.unwrap_err().contains("already exists"));
    
    // Index should still have only one node
    assert_eq!(index.len(), 1);
}

#[test]
fn test_insert_without_document() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    let embedding = vec![1.0, 2.0, 3.0];
    let result = index.insert("node1".to_string(), embedding, None);
    
    assert!(result.is_ok());
    assert_eq!(index.len(), 1);
    
    let node = index.get_node("node1").unwrap();
    assert!(node.document.is_none());
}

#[test]
fn test_insert_different_embedding_dimensions() {
    let mut index = HNSWIndex::new(5, 16, 200);
    
    // Insert nodes with different embedding dimensions
    let embeddings = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0],
        vec![-1.0, -2.0, -3.0, -4.0, -5.0],
    ];
    
    for (i, embedding) in embeddings.iter().enumerate() {
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("content_{}", i),
        };
        
        let result = index.insert(format!("node_{}", i), embedding.clone(), Some(doc));
        assert!(result.is_ok());
    }
    
    assert_eq!(index.len(), 3);
}

#[test]
fn test_select_neighbors() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert some nodes to create a small graph
    for i in 0..10 {
        let embedding = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
        index.insert(format!("node_{}", i), embedding, None).unwrap();
    }
    
    // Create some candidate nodes with distances
    let candidates = vec![
        (0, 1.0),
        (1, 2.0),
        (2, 3.0),
        (3, 4.0),
        (4, 5.0),
    ];
    
    // Select 3 neighbors
    let selected = index.select_neighbors(&candidates, 3);
    
    // Should select 3 neighbors
    assert_eq!(selected.len(), 3);
    
    // Should include the closest neighbor (index 0)
    assert!(selected.contains(&0));
}

#[test]
fn test_select_neighbors_less_candidates_than_m() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert some nodes
    for i in 0..5 {
        let embedding = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
        index.insert(format!("node_{}", i), embedding, None).unwrap();
    }
    
    let candidates = vec![(0, 1.0), (1, 2.0)];
    
    // Select 5 neighbors but only 2 candidates available
    let selected = index.select_neighbors(&candidates, 5);
    
    // Should select all available candidates
    assert_eq!(selected.len(), 2);
    assert!(selected.contains(&0));
    assert!(selected.contains(&1));
}

#[test]
fn test_rebalance() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert some nodes
    for i in 0..10 {
        let embedding = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
        index.insert(format!("node_{}", i), embedding, None).unwrap();
    }
    
    // Rebalance should succeed
    let result = index.rebalance();
    assert!(result.is_ok());
    
    // Index should still have all nodes
    assert_eq!(index.len(), 10);
}

#[test]
fn test_rebalance_empty_index() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Rebalance empty index should succeed
    let result = index.rebalance();
    assert!(result.is_ok());
    
    // Index should still be empty
    assert!(index.is_empty());
}

#[test]
fn test_insert_large_dataset() {
    let mut index = HNSWIndex::new(10, 16, 200);
    
    // Insert 100 nodes
    for i in 0..100 {
        let embedding: Vec<f32> = (0..10).map(|j| (i + j) as f32).collect();
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("content_{}", i),
        };
        
        let result = index.insert(format!("node_{}", i), embedding, Some(doc));
        assert!(result.is_ok());
    }
    
    assert_eq!(index.len(), 100);
    
    // Verify all nodes are accessible
    for i in 0..100 {
        let node = index.get_node(&format!("node_{}", i));
        assert!(node.is_some());
        assert_eq!(node.unwrap().id, format!("node_{}", i));
    }
}

#[test]
fn test_insert_with_zero_embedding() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    let zero_embedding = vec![0.0, 0.0, 0.0];
    let doc = Document {
        id: "zero_doc".to_string(),
        data: "zero content".to_string(),
    };
    
    let result = index.insert("zero_node".to_string(), zero_embedding, Some(doc));
    assert!(result.is_ok());
    
    let node = index.get_node("zero_node").unwrap();
    assert_eq!(node.embedding, vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_insert_with_negative_embeddings() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    let negative_embedding = vec![-1.0, -2.0, -3.0];
    let doc = Document {
        id: "negative_doc".to_string(),
        data: "negative content".to_string(),
    };
    
    let result = index.insert("negative_node".to_string(), negative_embedding, Some(doc));
    assert!(result.is_ok());
    
    let node = index.get_node("negative_node").unwrap();
    assert_eq!(node.embedding, vec![-1.0, -2.0, -3.0]);
}
