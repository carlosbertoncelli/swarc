use std::collections::HashMap;
use crate::types::HNSWNode;

/// HNSW Index structure
#[derive(Debug)]
pub struct HNSWIndex<T> {
    pub(crate) nodes: Vec<HNSWNode<T>>,
    pub(crate) node_id_to_index: HashMap<String, usize>,
    pub(crate) max_layers: usize,
    pub m: usize, // maximum number of connections per node
    pub m_max: usize, // maximum number of connections for layer 0
    pub ef_construction: usize, // size of dynamic candidate list
    pub(crate) ml: f32, // normalization factor for level generation
    pub entry_point: Option<usize>, // index of entry point
}

impl<T> HNSWIndex<T> {
    /// Create a new HNSW index
    pub fn new(_dim: usize, m: usize, ef_construction: usize) -> Self {
        let max_layers = (f32::ln(1000.0) / f32::ln(2.0)) as usize + 1; // reasonable default
        let m_max = m;
        let ml = 1.0 / f32::ln(2.0);
        
        Self {
            nodes: Vec::new(),
            node_id_to_index: HashMap::new(),
            max_layers,
            m,
            m_max,
            ef_construction,
            ml,
            entry_point: None,
        }
    }

    /// Calculate Euclidean distance between two vectors
    pub fn distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Generate random level for a new node
    pub fn generate_level(&self) -> usize {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let uniform = rng.gen::<f32>();
        (-f32::ln(uniform) * self.ml) as usize
    }

    /// Get the number of nodes in the index
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get a reference to a node by ID
    pub fn get_node(&self, id: &str) -> Option<&HNSWNode<T>> {
        self.node_id_to_index.get(id).and_then(|&idx| self.nodes.get(idx))
    }

    /// Get all node IDs in the index
    pub fn get_all_ids(&self) -> Vec<String> {
        self.nodes.iter().map(|node| node.id.clone()).collect()
    }
}
