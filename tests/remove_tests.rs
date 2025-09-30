use swarc::index::HNSWIndex;
use swarc::types::Document;

#[test]
fn test_remove_single_node() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    let doc = Document {
        id: "test_doc".to_string(),
        data: "test content".to_string(),
    };
    
    let embedding = vec![1.0, 2.0, 3.0];
    index.insert("node1".to_string(), embedding, Some(doc.clone())).unwrap();
    
    assert_eq!(index.len(), 1);
    assert!(index.contains("node1"));
    
    let removed_doc = index.remove("node1").unwrap();
    
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
    assert!(!index.contains("node1"));
    assert!(removed_doc.is_some());
    assert_eq!(removed_doc.unwrap().id, "test_doc");
}

#[test]
fn test_remove_nonexistent_node() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    let result = index.remove("nonexistent");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

#[test]
fn test_remove_from_empty_index() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    let result = index.remove("any_node");
    assert!(result.is_err());
}

#[test]
fn test_remove_multiple_nodes() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert multiple nodes
    for i in 0..5 {
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("content_{}", i),
        };
        let embedding = vec![i as f32, 0.0, 0.0];
        index.insert(format!("node_{}", i), embedding, Some(doc)).unwrap();
    }
    
    assert_eq!(index.len(), 5);
    
    // Remove multiple nodes
    let ids = ["node_1", "node_3", "node_4"];
    let removed_docs = index.remove_multiple(&ids).unwrap();
    
    assert_eq!(index.len(), 2);
    assert_eq!(removed_docs.len(), 3);
    
    // Check remaining nodes
    assert!(index.contains("node_0"));
    assert!(index.contains("node_2"));
    assert!(!index.contains("node_1"));
    assert!(!index.contains("node_3"));
    assert!(!index.contains("node_4"));
}

#[test]
fn test_remove_multiple_with_nonexistent() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert one node
    let doc = Document {
        id: "doc_0".to_string(),
        data: "content_0".to_string(),
    };
    let embedding = vec![0.0, 0.0, 0.0];
    index.insert("node_0".to_string(), embedding, Some(doc)).unwrap();
    
    // Try to remove multiple nodes including nonexistent ones
    let ids = ["node_0", "nonexistent", "node_1"];
    let result = index.remove_multiple(&ids);
    
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
    
    // Original node should still be there
    assert_eq!(index.len(), 1);
    assert!(index.contains("node_0"));
}

#[test]
fn test_clear_index() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert multiple nodes
    for i in 0..10 {
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("content_{}", i),
        };
        let embedding = vec![i as f32, 0.0, 0.0];
        index.insert(format!("node_{}", i), embedding, Some(doc)).unwrap();
    }
    
    assert_eq!(index.len(), 10);
    assert!(!index.is_empty());
    
    // Clear the index
    index.clear();
    
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
    assert!(index.get_all_ids().is_empty());
}

#[test]
fn test_clear_empty_index() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    assert!(index.is_empty());
    
    // Clear empty index should work
    index.clear();
    
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
}

#[test]
fn test_contains() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Initially empty
    assert!(!index.contains("node1"));
    
    // Insert a node
    let doc = Document {
        id: "test_doc".to_string(),
        data: "test content".to_string(),
    };
    let embedding = vec![1.0, 2.0, 3.0];
    index.insert("node1".to_string(), embedding, Some(doc)).unwrap();
    
    // Should contain the node
    assert!(index.contains("node1"));
    assert!(!index.contains("node2"));
    
    // Remove the node
    index.remove("node1").unwrap();
    
    // Should no longer contain the node
    assert!(!index.contains("node1"));
}

#[test]
fn test_remove_and_reinsert() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    let doc1 = Document {
        id: "doc1".to_string(),
        data: "content1".to_string(),
    };
    let embedding1 = vec![1.0, 2.0, 3.0];
    
    // Insert node
    index.insert("node1".to_string(), embedding1.clone(), Some(doc1.clone())).unwrap();
    assert_eq!(index.len(), 1);
    
    // Remove node
    let removed_doc = index.remove("node1").unwrap();
    assert_eq!(index.len(), 0);
    assert!(removed_doc.is_some());
    
    // Reinsert with same ID
    let doc2 = Document {
        id: "doc2".to_string(),
        data: "content2".to_string(),
    };
    index.insert("node1".to_string(), embedding1, Some(doc2)).unwrap();
    assert_eq!(index.len(), 1);
    assert!(index.contains("node1"));
}

#[test]
fn test_remove_node_without_document() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    let embedding = vec![1.0, 2.0, 3.0];
    index.insert("node1".to_string(), embedding, None).unwrap();
    
    assert_eq!(index.len(), 1);
    
    let removed_doc = index.remove("node1").unwrap();
    
    assert_eq!(index.len(), 0);
    assert!(removed_doc.is_none());
}

#[test]
fn test_remove_affects_search() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert multiple nodes
    for i in 0..5 {
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("content_{}", i),
        };
        let embedding = vec![i as f32, 0.0, 0.0];
        index.insert(format!("node_{}", i), embedding, Some(doc)).unwrap();
    }
    
    let query = vec![0.0, 0.0, 0.0];
    let results_before = index.search(&query, 5);
    assert_eq!(results_before.len(), 5);
    
    // Remove a node
    index.remove("node_2").unwrap();
    
    // Search again
    let results_after = index.search(&query, 5);
    assert_eq!(results_after.len(), 4);
    
    // The removed node should not be in results
    for (id, _distance, _doc) in results_after {
        assert_ne!(id, "node_2");
    }
}

#[test]
fn test_remove_multiple_affects_search() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert multiple nodes
    for i in 0..10 {
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("content_{}", i),
        };
        let embedding = vec![i as f32, 0.0, 0.0];
        index.insert(format!("node_{}", i), embedding, Some(doc)).unwrap();
    }
    
    let query = vec![5.0, 0.0, 0.0];
    let results_before = index.search(&query, 10);
    assert_eq!(results_before.len(), 10);
    
    // Remove multiple nodes
    let ids = ["node_1", "node_3", "node_5", "node_7"];
    index.remove_multiple(&ids).unwrap();
    
    // Search again
    let results_after = index.search(&query, 10);
    assert_eq!(results_after.len(), 6);
    
    // The removed nodes should not be in results
    for (id, _distance, _doc) in results_after {
        assert!(!ids.contains(&id.as_str()));
    }
}

#[test]
fn test_remove_entry_point() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert nodes
    for i in 0..5 {
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("content_{}", i),
        };
        let embedding = vec![i as f32, 0.0, 0.0];
        index.insert(format!("node_{}", i), embedding, Some(doc)).unwrap();
    }
    
    // Remove the first node (likely the entry point)
    index.remove("node_0").unwrap();
    
    // Index should still be functional
    assert_eq!(index.len(), 4);
    
    // Search should still work
    let query = vec![2.0, 0.0, 0.0];
    let results = index.search(&query, 3);
    assert_eq!(results.len(), 3);
}

#[test]
fn test_remove_all_nodes() {
    let mut index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    // Insert nodes
    for i in 0..3 {
        let doc = Document {
            id: format!("doc_{}", i),
            data: format!("content_{}", i),
        };
        let embedding = vec![i as f32, 0.0, 0.0];
        index.insert(format!("node_{}", i), embedding, Some(doc)).unwrap();
    }
    
    // Remove all nodes one by one
    for i in 0..3 {
        index.remove(&format!("node_{}", i)).unwrap();
    }
    
    // Index should be empty
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    
    // Search should return empty results
    let query = vec![0.0, 0.0, 0.0];
    let results = index.search(&query, 5);
    assert!(results.is_empty());
}
