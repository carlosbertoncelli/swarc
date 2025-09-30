use swarc::types::{Document, HNSWNode};

#[test]
fn test_document_creation() {
    let doc = Document {
        id: "test_doc".to_string(),
        data: "test content".to_string(),
    };
    
    assert_eq!(doc.id, "test_doc");
    assert_eq!(doc.data, "test content");
}

#[test]
fn test_document_with_different_types() {
    let doc_string = Document {
        id: "string_doc".to_string(),
        data: "string content".to_string(),
    };
    
    let doc_number = Document {
        id: "number_doc".to_string(),
        data: 42,
    };
    
    let doc_vector = Document {
        id: "vector_doc".to_string(),
        data: vec![1, 2, 3],
    };
    
    assert_eq!(doc_string.data, "string content");
    assert_eq!(doc_number.data, 42);
    assert_eq!(doc_vector.data, vec![1, 2, 3]);
}

#[test]
fn test_hnsw_node_creation() {
    let embedding = vec![1.0, 2.0, 3.0];
    let doc = Document {
        id: "test_doc".to_string(),
        data: "test content".to_string(),
    };
    
    let node = HNSWNode::new("node1".to_string(), embedding.clone(), Some(doc.clone()));
    
    assert_eq!(node.id, "node1");
    assert_eq!(node.embedding, embedding);
    assert_eq!(node.document.as_ref().unwrap().id, "test_doc");
    assert_eq!(node.document.as_ref().unwrap().data, "test content");
    assert!(node.connections.is_empty());
}

#[test]
fn test_hnsw_node_without_document() {
    let embedding = vec![1.0, 2.0, 3.0];
    let node: HNSWNode<String> = HNSWNode::new("node1".to_string(), embedding.clone(), None);
    
    assert_eq!(node.id, "node1");
    assert_eq!(node.embedding, embedding);
    assert!(node.document.is_none());
    assert!(node.connections.is_empty());
}

#[test]
fn test_hnsw_node_clone() {
    let embedding = vec![1.0, 2.0, 3.0];
    let doc = Document {
        id: "test_doc".to_string(),
        data: "test content".to_string(),
    };
    
    let node1 = HNSWNode::new("node1".to_string(), embedding.clone(), Some(doc.clone()));
    let node2 = node1.clone();
    
    assert_eq!(node1.id, node2.id);
    assert_eq!(node1.embedding, node2.embedding);
    assert_eq!(node1.document.as_ref().unwrap().id, node2.document.as_ref().unwrap().id);
    assert_eq!(node1.document.as_ref().unwrap().data, node2.document.as_ref().unwrap().data);
}

#[test]
fn test_document_serialization() {
    use serde_json;
    
    let doc = Document {
        id: "test_doc".to_string(),
        data: "test content".to_string(),
    };
    
    let serialized = serde_json::to_string(&doc).unwrap();
    let deserialized: Document<String> = serde_json::from_str(&serialized).unwrap();
    
    assert_eq!(doc.id, deserialized.id);
    assert_eq!(doc.data, deserialized.data);
}

#[test]
fn test_document_with_complex_data() {
    #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq, Clone)]
    struct ComplexData {
        name: String,
        values: Vec<i32>,
        metadata: std::collections::HashMap<String, String>,
    }
    
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("key1".to_string(), "value1".to_string());
    metadata.insert("key2".to_string(), "value2".to_string());
    
    let complex_data = ComplexData {
        name: "test".to_string(),
        values: vec![1, 2, 3],
        metadata,
    };
    
    let doc = Document {
        id: "complex_doc".to_string(),
        data: complex_data.clone(),
    };
    
    assert_eq!(doc.id, "complex_doc");
    assert_eq!(doc.data, complex_data);
}
