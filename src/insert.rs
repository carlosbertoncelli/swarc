use crate::index::HNSWIndex;
use crate::types::{HNSWNode, Document};
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone)]
struct IndexSnapshot<T> {
    nodes: Vec<HNSWNode<T>>,
    node_id_to_index: HashMap<String, usize>,
    entry_point: Option<usize>,
    max_layers: usize,
    m: usize,
    m_max: usize,
    ef_construction: usize,
}

#[derive(Clone)]
struct ConnectionUpdate {
    node_index: usize,
    layer: usize,
    neighbors: Vec<usize>,
    neighbor_updates: Vec<(usize, usize, usize)>, // (neighbor_index, layer, new_connection)
}

impl<T: Clone + Send + Sync> HNSWIndex<T> {
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
                        let dist = self.distance(&selected_node.embedding, &candidate_node.embedding);
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

    /// Insert multiple nodes in parallel
    /// This method processes insertions in parallel while maintaining thread safety
    pub fn insert_parallel(&mut self, items: Vec<(String, Vec<f32>, Option<Document<T>>)>) -> Result<Vec<Result<(), String>>, String> {
        if items.is_empty() {
            return Ok(vec![]);
        }

        // Check for duplicate IDs first
        let mut seen_ids = std::collections::HashSet::new();
        for (id, _, _) in &items {
            if !seen_ids.insert(id) {
                return Err(format!("Duplicate ID found: '{}'", id));
            }
            if self.node_id_to_index.contains_key(id) {
                return Err(format!("Node with id '{}' already exists in index", id));
            }
        }

        // Generate levels for all items first
        let items_with_levels: Vec<_> = items.into_iter()
            .map(|(id, embedding, document)| {
                let level = self.generate_level();
                (id, embedding, document, level)
            })
            .collect();

        // Sort by level (descending) to insert higher-level nodes first
        let mut sorted_items = items_with_levels;
        sorted_items.sort_by(|a, b| b.3.cmp(&a.3));

        // Use parallel processing for the actual insertion
        let results = self.insert_parallel_batch(sorted_items)?;

        Ok(results)
    }

    /// Insert a batch of nodes in parallel using true parallelism
    fn insert_parallel_batch(&mut self, items: Vec<(String, Vec<f32>, Option<Document<T>>, usize)>) -> Result<Vec<Result<(), String>>, String> {
        if items.is_empty() {
            return Ok(vec![]);
        }

        // For true parallelism, we'll use a simpler approach:
        // 1. Pre-allocate all nodes
        // 2. Process insertions in parallel using thread-safe operations
        // 3. Apply results sequentially to maintain graph integrity

        let num_items = items.len();
        let mut results = Vec::new();

        // Step 1: Pre-allocate all nodes
        let mut new_nodes = Vec::with_capacity(num_items);
        let mut node_id_to_index = HashMap::new();

        for (i, (id, embedding, document, level)) in items.iter().enumerate() {
            let node_index = self.nodes.len() + i;
            let mut new_node = HNSWNode::new(id.clone(), embedding.clone(), document.as_ref().cloned());
            new_node.connections.resize(level + 1, Vec::new());
            new_nodes.push(new_node);
            node_id_to_index.insert(id.clone(), node_index);
        }

        // Step 2: Add all nodes to the index
        self.nodes.extend(new_nodes);
        self.node_id_to_index.extend(node_id_to_index);

        // Step 3: Process connections in parallel using rayon
        // We'll parallelize the computation of connections, but apply them sequentially
        let connection_tasks: Vec<_> = items
            .iter()
            .enumerate()
            .map(|(i, (id, embedding, _document, level))| {
                let node_index = self.nodes.len() - num_items + i;
                (node_index, id.clone(), embedding.clone(), *level)
            })
            .collect();

        // Use rayon to compute connections in parallel
        let connection_results: Vec<_> = connection_tasks
            .par_iter()
            .map(|(node_index, _id, embedding, level)| {
                self.compute_connections_for_parallel_insertion(*node_index, embedding, *level)
            })
            .collect();

        // Step 4: Apply connection results sequentially
        for (i, connection_result) in connection_results.into_iter().enumerate() {
            match connection_result {
                Ok(connections) => {
                    // Apply connections to the node
                    let node_index = self.nodes.len() - num_items + i;
                    if let Some(node) = self.nodes.get_mut(node_index) {
                        for (layer, layer_connections) in connections.into_iter().enumerate() {
                            if layer < node.connections.len() {
                                node.connections[layer] = layer_connections;
                            }
                        }
                    }
                    results.push(Ok(()));
                }
                Err(e) => {
                    results.push(Err(e));
                }
            }
        }

        // Step 5: Update entry point
        self.update_entry_point_after_batch_insertion(&items);

        Ok(results)
    }

    /// Build connections for a specific node (used in parallel processing)
    fn build_connections_for_node(&mut self, node_index: usize, embedding: &[f32], level: usize) -> Result<(), String> {
        if self.entry_point.is_none() {
            self.entry_point = Some(node_index);
            return Ok(());
        }

        let mut entry_points = vec![self.entry_point.unwrap()];

        // Search from top layer down to level + 1
        for layer in (level + 1..self.max_layers).rev() {
            if !entry_points.is_empty() {
                let candidates = self.search_layer(embedding, &entry_points, layer, 1);
                entry_points = candidates.into_iter().map(|(id, _)| id).collect();
            }
        }

        // Search and connect from level down to 0
        for layer in (0..=level).rev() {
            let m_layer = if layer == 0 { self.m_max } else { self.m };
            let candidates = self.search_layer(embedding, &entry_points, layer, self.ef_construction);
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

        Ok(())
    }

    /// Update entry point after batch insertion
    fn update_entry_point_after_batch_insertion(&mut self, items: &[(String, Vec<f32>, Option<Document<T>>, usize)]) {
        if items.is_empty() {
            return;
        }

        // Find the highest level among the newly inserted nodes
        let max_level = items.iter().map(|(_, _, _, level)| *level).max().unwrap_or(0);
        
        if max_level > 0 {
            // Find a node with the highest level to potentially become the new entry point
            let mut best_entry_point = self.entry_point;
            let mut best_level = 0;

            if let Some(ep) = self.entry_point {
                if let Some(ep_node) = self.nodes.get(ep) {
                    best_level = ep_node.connections.len().saturating_sub(1);
                }
            }

            // Check newly inserted nodes
            let start_index = self.nodes.len() - items.len();
            for i in start_index..self.nodes.len() {
                if let Some(node) = self.nodes.get(i) {
                    let node_level = node.connections.len().saturating_sub(1);
                    if node_level > best_level {
                        best_level = node_level;
                        best_entry_point = Some(i);
                    }
                }
            }

            self.entry_point = best_entry_point;
        }
    }

    /// Insert a single node with a pre-determined level
    /// This is used internally by the parallel insertion process
    fn insert_single_with_level(&mut self, id: String, embedding: Vec<f32>, document: Option<Document<T>>, level: usize) -> Result<(), String> {
        if self.node_id_to_index.contains_key(&id) {
            return Err(format!("Node with id '{}' already exists", id));
        }
        
        let node_index = self.nodes.len();
        
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

    /// Insert multiple nodes sequentially (for comparison with parallel version)
    pub fn insert_multiple(&mut self, items: Vec<(String, Vec<f32>, Option<Document<T>>)>) -> Result<Vec<Result<(), String>>, String> {
        let mut results = Vec::new();
        
        for (id, embedding, document) in items {
            let result = self.insert(id, embedding, document);
            results.push(result);
        }
        
        Ok(results)
    }

    /// Compute connections for parallel insertion (thread-safe)
    fn compute_connections_for_parallel_insertion(&self, _node_index: usize, embedding: &[f32], level: usize) -> Result<Vec<Vec<usize>>, String> {
        if self.entry_point.is_none() {
            return Ok(vec![]);
        }

        let mut entry_points = vec![self.entry_point.unwrap()];
        let mut all_connections = Vec::new();

        // Search from top layer down to level + 1
        for layer in (level + 1..self.max_layers).rev() {
            if !entry_points.is_empty() {
                let candidates = self.search_layer(embedding, &entry_points, layer, 1);
                entry_points = candidates.into_iter().map(|(id, _)| id).collect();
            }
        }

        // Search and connect from level down to 0
        for layer in (0..=level).rev() {
            let m_layer = if layer == 0 { self.m_max } else { self.m };
            let candidates = self.search_layer(embedding, &entry_points, layer, self.ef_construction);
            let neighbors = self.select_neighbors(&candidates, m_layer);

            all_connections.push(neighbors.clone());
            entry_points = neighbors;
        }

        Ok(all_connections)
    }

    /// Compute connections for a node in parallel using a snapshot
    fn compute_connections_parallel(
        &self,
        snapshot: &IndexSnapshot<T>,
        node_index: usize,
        _id: &str,
        embedding: &[f32],
        _document: Option<&Document<T>>,
        level: usize,
    ) -> Result<Vec<ConnectionUpdate>, String> {
        if snapshot.entry_point.is_none() {
            return Ok(vec![]);
        }

        let mut entry_points = vec![snapshot.entry_point.unwrap()];
        let mut updates = Vec::new();

        // Search from top layer down to level + 1
        for layer in (level + 1..snapshot.max_layers).rev() {
            if !entry_points.is_empty() {
                let candidates = self.search_layer_snapshot(snapshot, embedding, &entry_points, layer, 1);
                entry_points = candidates.into_iter().map(|(id, _)| id).collect();
            }
        }

        // Search and connect from level down to 0
        for layer in (0..=level).rev() {
            let m_layer = if layer == 0 { snapshot.m_max } else { snapshot.m };
            let candidates = self.search_layer_snapshot(snapshot, embedding, &entry_points, layer, snapshot.ef_construction);
            let neighbors = self.select_neighbors_snapshot(snapshot, &candidates, m_layer);

            // Create connection update
            let mut neighbor_updates = Vec::new();
            for &neighbor_id in &neighbors {
                neighbor_updates.push((neighbor_id, layer, node_index));
            }

            updates.push(ConnectionUpdate {
                node_index,
                layer,
                neighbors: neighbors.clone(),
                neighbor_updates,
            });

            // Update entry points for next layer
            entry_points = neighbors;
        }

        Ok(updates)
    }

    /// Search layer using a snapshot
    fn search_layer_snapshot(
        &self,
        snapshot: &IndexSnapshot<T>,
        query: &[f32],
        entry_points: &[usize],
        layer: usize,
        k: usize,
    ) -> Vec<(usize, f32)> {
        let mut candidates = Vec::new();
        let mut visited = std::collections::HashSet::new();

        // Initialize candidates with entry points
        for &ep in entry_points {
            if let Some(node) = snapshot.nodes.get(ep) {
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

            if let Some(node) = snapshot.nodes.get(current_id) {
                let neighbors = if layer == 0 {
                    (0..snapshot.nodes.len()).filter(|&i| i != current_id).collect::<Vec<_>>()
                } else if layer < node.connections.len() {
                    node.connections[layer].clone()
                } else {
                    continue;
                };

                for neighbor_id in neighbors {
                    if !visited.contains(&neighbor_id) {
                        visited.insert(neighbor_id);

                        if let Some(neighbor) = snapshot.nodes.get(neighbor_id) {
                            let dist = self.distance(query, &neighbor.embedding);

                            if candidates.len() < k || dist < candidates.last().unwrap().1 {
                                candidates.push((neighbor_id, dist));
                                candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                                if candidates.len() > k {
                                    candidates.pop();
                                }

                                dynamic_candidates.push((neighbor_id, dist));
                                dynamic_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                                if dynamic_candidates.len() > snapshot.ef_construction {
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

    /// Select neighbors using a snapshot
    fn select_neighbors_snapshot(&self, snapshot: &IndexSnapshot<T>, candidates: &[(usize, f32)], m: usize) -> Vec<usize> {
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
                        (snapshot.nodes.get(selected_id), snapshot.nodes.get(*candidate_id)) {
                        let dist = self.distance(&selected_node.embedding, &candidate_node.embedding);
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

    /// Apply a connection update to the index
    fn apply_connection_update(&mut self, update: ConnectionUpdate) {
        // Update the new node's connections
        if let Some(new_node) = self.nodes.get_mut(update.node_index) {
            if update.layer < new_node.connections.len() {
                new_node.connections[update.layer] = update.neighbors;
            }
        }

        // Update neighbor connections
        for (neighbor_id, layer, new_connection) in update.neighbor_updates {
            if let Some(neighbor) = self.nodes.get_mut(neighbor_id) {
                if layer < neighbor.connections.len() {
                    neighbor.connections[layer].push(new_connection);
                }
            }
        }
    }
}
