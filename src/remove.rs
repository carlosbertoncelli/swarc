use crate::index::HNSWIndex;
use crate::types::Document;

impl<T> HNSWIndex<T> {
    /// Remove a node from the index
    pub fn remove(&mut self, id: &str) -> Result<Option<Document<T>>, String> {
        let node_index = match self.node_id_to_index.remove(id) {
            Some(idx) => idx,
            None => return Err(format!("Node with id '{}' not found", id)),
        };
        
        let removed_node = self.nodes.remove(node_index);
        
        // Update node_id_to_index for remaining nodes
        for (_, idx) in self.node_id_to_index.iter_mut() {
            if *idx > node_index {
                *idx -= 1;
            }
        }
        
        // Remove connections to the removed node
        for node in &mut self.nodes {
            for layer_connections in &mut node.connections {
                layer_connections.retain(|&x| x != node_index);
                // Update indices for nodes that came after the removed node
                for connection in layer_connections.iter_mut() {
                    if *connection > node_index {
                        *connection -= 1;
                    }
                }
            }
        }
        
        // Update entry point if it was the removed node
        if self.entry_point == Some(node_index) {
            self.entry_point = if self.nodes.is_empty() {
                None
            } else {
                // Find a new entry point - prefer nodes with higher levels
                let mut best_entry_point = 0;
                let mut max_level = 0;
                
                for (i, node) in self.nodes.iter().enumerate() {
                    let level = node.connections.len().saturating_sub(1);
                    if level > max_level {
                        max_level = level;
                        best_entry_point = i;
                    }
                }
                
                Some(best_entry_point)
            };
        } else if let Some(ep) = self.entry_point {
            if ep > node_index {
                self.entry_point = Some(ep - 1);
            }
        }
        
        Ok(removed_node.document)
    }

    /// Remove multiple nodes by their IDs
    pub fn remove_multiple(&mut self, ids: &[&str]) -> Result<Vec<Option<Document<T>>>, String> {
        // First, verify all nodes exist before removing any
        for id in ids {
            if !self.node_id_to_index.contains_key(*id) {
                return Err(format!("Node with id '{}' not found", id));
            }
        }
        
        // Now remove all nodes
        let mut removed_docs = Vec::new();
        for id in ids {
            match self.remove(id) {
                Ok(doc) => removed_docs.push(doc),
                Err(e) => return Err(e), // This shouldn't happen since we verified above
            }
        }
        
        Ok(removed_docs)
    }

    /// Clear all nodes from the index
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.node_id_to_index.clear();
        self.entry_point = None;
    }

    /// Check if a node exists in the index
    pub fn contains(&self, id: &str) -> bool {
        self.node_id_to_index.contains_key(id)
    }
}
