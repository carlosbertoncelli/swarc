use serde::{Deserialize, Serialize};

/// Distance metric types supported by the HNSW index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
}

/// Trait for distance calculations between vectors
pub trait Distance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
}

impl Distance for DistanceMetric {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::Euclidean => {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt()
            }
            DistanceMetric::Cosine => {
                let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                
                if norm_a == 0.0 || norm_b == 0.0 {
                    0.0
                } else {
                    1.0 - (dot_product / (norm_a * norm_b))
                }
            }
        }
    }
}

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
