use swarc::index::HNSWIndex;
use swarc::types::Document;

#[test]
fn test_index_creation() {
    let index: HNSWIndex<String> = HNSWIndex::new(128, 16, 200);
    
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
    assert!(index.entry_point.is_none());
    assert_eq!(index.m, 16);
    assert_eq!(index.m_max, 16);
    assert_eq!(index.ef_construction, 200);
}

#[test]
fn test_index_creation_with_different_parameters() {
    let index1: HNSWIndex<String> = HNSWIndex::new(64, 8, 100);
    let index2: HNSWIndex<String> = HNSWIndex::new(256, 32, 500);
    
    assert_eq!(index1.m, 8);
    assert_eq!(index1.ef_construction, 100);
    
    assert_eq!(index2.m, 32);
    assert_eq!(index2.ef_construction, 500);
}

#[test]
fn test_distance_calculation() {
    let _index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    let a = vec![0.0, 0.0, 0.0];
    let b = vec![3.0, 4.0, 0.0];
    let c = vec![1.0, 1.0, 1.0];
    
    let dist_ab = HNSWIndex::<String>::distance(&a, &b);
    let dist_ac = HNSWIndex::<String>::distance(&a, &c);
    let dist_bc = HNSWIndex::<String>::distance(&b, &c);
    
    // Distance from (0,0,0) to (3,4,0) should be 5
    assert!((dist_ab - 5.0).abs() < 1e-6);
    
    // Distance from (0,0,0) to (1,1,1) should be sqrt(3)
    assert!((dist_ac - 3.0_f32.sqrt()).abs() < 1e-6);
    
    // Distance from (3,4,0) to (1,1,1) should be sqrt(14)
    assert!((dist_bc - 14.0_f32.sqrt()).abs() < 1e-6);
}

#[test]
fn test_distance_symmetry() {
    let _index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    
    let dist_ab = HNSWIndex::<String>::distance(&a, &b);
    let dist_ba = HNSWIndex::<String>::distance(&b, &a);
    
    assert!((dist_ab - dist_ba).abs() < 1e-6);
}

#[test]
fn test_distance_zero_vectors() {
    let _index: HNSWIndex<String> = HNSWIndex::new(3, 16, 200);
    
    let zero = vec![0.0, 0.0, 0.0];
    let a = vec![1.0, 2.0, 3.0];
    
    let dist_zero_zero = HNSWIndex::<String>::distance(&zero, &zero);
    let dist_zero_a = HNSWIndex::<String>::distance(&zero, &a);
    
    assert!((dist_zero_zero - 0.0).abs() < 1e-6);
    assert!((dist_zero_a - 14.0_f32.sqrt()).abs() < 1e-6);
}

#[test]
fn test_level_generation() {
    let index: HNSWIndex<String> = HNSWIndex::new(128, 16, 200);
    
    // Generate multiple levels and check they're reasonable
    let levels: Vec<usize> = (0..100).map(|_| index.generate_level()).collect();
    
    // All levels should be non-negative
    for level in &levels {
        assert!(*level >= 0); // This is always true for usize, but kept for clarity
    }
    
    // Most levels should be small (exponential distribution)
    let small_levels = levels.iter().filter(|&&l| l < 10).count();
    assert!(small_levels > 80); // Most levels should be small
}

#[test]
fn test_index_utility_methods() {
    let mut index = HNSWIndex::new(128, 16, 200);
    
    // Initially empty
    assert!(index.is_empty());
    assert_eq!(index.len(), 0);
    assert!(index.get_all_ids().is_empty());
    
    // Insert a node
    let doc = Document {
        id: "test_doc".to_string(),
        data: "test content".to_string(),
    };
    
    let embedding = vec![1.0, 2.0, 3.0];
    index.insert("node1".to_string(), embedding, Some(doc)).unwrap();
    
    // Check state after insertion
    assert!(!index.is_empty());
    assert_eq!(index.len(), 1);
    assert_eq!(index.get_all_ids(), vec!["node1"]);
    
    // Check get_node
    let node = index.get_node("node1");
    assert!(node.is_some());
    assert_eq!(node.unwrap().id, "node1");
    
    // Check non-existent node
    let non_existent = index.get_node("non_existent");
    assert!(non_existent.is_none());
}

#[test]
fn test_index_with_different_data_types() {
    // Test with String data
    let mut index_string = HNSWIndex::new(3, 16, 200);
    let doc_string = Document {
        id: "string_doc".to_string(),
        data: "string content".to_string(),
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
    
    // All should work
    assert_eq!(index_string.len(), 1);
    assert_eq!(index_number.len(), 1);
    assert_eq!(index_vector.len(), 1);
}
