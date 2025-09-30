use serde::{Deserialize, Serialize};

/// Document structure for linking external data with embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document<T> {
    pub id: String,
    pub data: T,
}

/// HNSW Node containing embedding vector and connections
#[derive(Debug, Clone)]
pub struct HNSWNode<T> {
    pub id: String,
    pub embedding: Vec<f32>,
    pub document: Option<Document<T>>,
    pub connections: Vec<Vec<usize>>, // connections for each layer
}

impl<T> HNSWNode<T> {
    pub fn new(id: String, embedding: Vec<f32>, document: Option<Document<T>>) -> Self {
        Self {
            id,
            embedding,
            document,
            connections: Vec::new(),
        }
    }
}
