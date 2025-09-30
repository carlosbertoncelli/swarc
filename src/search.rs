use std::collections::HashSet;
use crate::index::HNSWIndex;
use crate::types::Document;

impl<T> HNSWIndex<T> {
    /// Search for nearest neighbors in a specific layer
    pub(crate) fn search_layer(&self, query: &[f32], entry_points: &[usize], layer: usize, k: usize) -> Vec<(usize, f32)> {
        let mut candidates = Vec::new();
        let mut visited = HashSet::new();
        
        // Initialize candidates with entry points
        for &ep in entry_points {
            if let Some(node) = self.nodes.get(ep) {
                // For layer 0, all nodes should be accessible
                if layer == 0 || layer < node.connections.len() {
                    let dist = self.distance(query, &node.embedding);
                    candidates.push((ep, dist));
                    visited.insert(ep);
                }
            }
        }
        
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let mut dynamic_candidates = candidates.clone();
        
        while !dynamic_candidates.is_empty() {
            let (current_id, _) = dynamic_candidates.remove(0);
            
            if let Some(node) = self.nodes.get(current_id) {
                // For layer 0, check all nodes; for higher layers, check connections
                let neighbors = if layer == 0 {
                    // In layer 0, we can consider all nodes as potential neighbors
                    (0..self.nodes.len()).filter(|&i| i != current_id).collect::<Vec<_>>()
                } else if layer < node.connections.len() {
                    node.connections[layer].clone()
                } else {
                    continue;
                };
                
                for neighbor_id in neighbors {
                    if !visited.contains(&neighbor_id) {
                        visited.insert(neighbor_id);
                        
                        if let Some(neighbor) = self.nodes.get(neighbor_id) {
                            let dist = self.distance(query, &neighbor.embedding);
                            
                            if candidates.len() < k || dist < candidates.last().unwrap().1 {
                                candidates.push((neighbor_id, dist));
                                candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                                
                                if candidates.len() > k {
                                    candidates.pop();
                                }
                                
                                dynamic_candidates.push((neighbor_id, dist));
                                dynamic_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                                
                                if dynamic_candidates.len() > self.ef_construction {
                                    dynamic_candidates.pop();
                                }
                            }
                        }
                    }
                }
            }
        }
        
        candidates
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32, Option<&Document<T>>)> {
        if self.entry_point.is_none() || self.nodes.is_empty() {
            return Vec::new();
        }
        
        let mut entry_points = vec![self.entry_point.unwrap()];
        
        // Find the highest layer that has nodes
        let max_existing_layer = self.nodes.iter()
            .map(|node| node.connections.len().saturating_sub(1))
            .max()
            .unwrap_or(0);
        
        // Search from the highest existing layer down to layer 1
        for layer in (1..=max_existing_layer).rev() {
            if !entry_points.is_empty() {
                let candidates = self.search_layer(query, &entry_points, layer, 1);
                entry_points = candidates.into_iter().map(|(id, _)| id).collect();
            }
        }
        
        // If no higher layers exist, start from entry point
        if max_existing_layer == 0 {
            entry_points = vec![self.entry_point.unwrap()];
        }
        
        // Search in layer 0
        let candidates = self.search_layer(query, &entry_points, 0, k);
        
        candidates.into_iter().map(|(id, dist)| {
            let node = &self.nodes[id];
            (node.id.clone(), dist, node.document.as_ref())
        }).collect()
    }
}
