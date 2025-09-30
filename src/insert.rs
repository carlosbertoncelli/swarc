use crate::index::HNSWIndex;
use crate::types::{HNSWNode, Document};

impl<T> HNSWIndex<T> {
    /// Select neighbors using the HNSW selection algorithm
    pub fn select_neighbors(&self, candidates: &[(usize, f32)], m: usize) -> Vec<usize> {
        if candidates.len() <= m {
            return candidates.iter().map(|(id, _)| *id).collect();
        }
        
        let mut selected = Vec::new();
        let mut remaining = candidates.to_vec();
        
        // Always select the closest neighbor
        if let Some((closest_id, _)) = remaining.first() {
            selected.push(*closest_id);
            remaining.remove(0);
        }
        
        while selected.len() < m && !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_score = f32::INFINITY;
            
            for (i, (candidate_id, candidate_dist)) in remaining.iter().enumerate() {
                let mut min_dist = f32::INFINITY;
                
                for &selected_id in &selected {
                    if let (Some(selected_node), Some(candidate_node)) = 
                        (self.nodes.get(selected_id), self.nodes.get(*candidate_id)) {
                        let dist = Self::distance(&selected_node.embedding, &candidate_node.embedding);
                        min_dist = min_dist.min(dist);
                    }
                }
                
                let score = *candidate_dist / min_dist;
                if score < best_score {
                    best_score = score;
                    best_idx = i;
                }
            }
            
            selected.push(remaining.remove(best_idx).0);
        }
        
        selected
    }

    /// Insert a new node into the index
    pub fn insert(&mut self, id: String, embedding: Vec<f32>, document: Option<Document<T>>) -> Result<(), String> {
        if self.node_id_to_index.contains_key(&id) {
            return Err(format!("Node with id '{}' already exists", id));
        }
        
        let node_index = self.nodes.len();
        let level = self.generate_level();
        
        let mut new_node = HNSWNode::new(id.clone(), embedding.clone(), document);
        new_node.connections.resize(level + 1, Vec::new());
        
        self.nodes.push(new_node);
        self.node_id_to_index.insert(id, node_index);
        
        if self.entry_point.is_none() {
            self.entry_point = Some(node_index);
            return Ok(());
        }
        
        let mut entry_points = vec![self.entry_point.unwrap()];
        
        // Search from top layer down to level + 1
        for layer in (level + 1..self.max_layers).rev() {
            if !entry_points.is_empty() {
                let candidates = self.search_layer(&embedding, &entry_points, layer, 1);
                entry_points = candidates.into_iter().map(|(id, _)| id).collect();
            }
        }
        
        // Search and connect from level down to 0
        for layer in (0..=level).rev() {
            let m_layer = if layer == 0 { self.m_max } else { self.m };
            let candidates = self.search_layer(&embedding, &entry_points, layer, self.ef_construction);
            let neighbors = self.select_neighbors(&candidates, m_layer);
            
            // Connect the new node to selected neighbors
            for &neighbor_id in &neighbors {
                if let Some(neighbor) = self.nodes.get_mut(neighbor_id) {
                    if layer < neighbor.connections.len() {
                        neighbor.connections[layer].push(node_index);
                    }
                }
            }
            
            // Connect neighbors to the new node
            if let Some(new_node) = self.nodes.get_mut(node_index) {
                new_node.connections[layer] = neighbors.clone();
            }
            
            // Update entry points for next layer
            entry_points = neighbors;
        }
        
        // Update entry point if new node is at a higher level
        if level > 0 {
            if let Some(ep) = self.entry_point {
                if let Some(ep_node) = self.nodes.get(ep) {
                    if level >= ep_node.connections.len() {
                        self.entry_point = Some(node_index);
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Rebalance the index by rebuilding connections
    /// This is a simplified rebalancing that can be extended for more sophisticated strategies
    pub fn rebalance(&mut self) -> Result<(), String> {
        if self.nodes.is_empty() {
            return Ok(());
        }

        // For now, this is a placeholder for rebalancing logic
        // In a full implementation, you might want to:
        // 1. Analyze connection distribution
        // 2. Rebuild connections for nodes with too many/few connections
        // 3. Reassign levels for better distribution
        // 4. Optimize entry point selection
        
        println!("Rebalancing index with {} nodes", self.nodes.len());
        
        // Simple validation: ensure all connections are bidirectional
        for (i, node) in self.nodes.iter().enumerate() {
            for (layer, connections) in node.connections.iter().enumerate() {
                for &neighbor_id in connections {
                    if let Some(neighbor) = self.nodes.get(neighbor_id) {
                        if layer < neighbor.connections.len() {
                            if !neighbor.connections[layer].contains(&i) {
                                println!("Warning: Non-bidirectional connection found between node {} and {}", i, neighbor_id);
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
}
