// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! HNSW implementation
//!

use std::collections::BTreeSet;
use std::collections::HashSet;

use num_traits::Float;

use super::graph::{InMemoryVectorStorage, VectorStorage};

mod builder;
mod storage;

#[derive(Debug, Eq)]
pub struct GraphNode {
    pub id: u32,
    pub neighbors: Vec<Vec<u32>>,
}

impl PartialEq for GraphNode {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl GraphNode {
    pub fn new(id: u32, neighbors: Vec<u32>) -> Self {
        Self {
            id,
            neighbors: vec![],
        }
    }
}

/// HNSW Graph
///
/// A sealed graph.
pub struct HNSW<T: Float, S: VectorStorage<T>> {
    vectors: S,

    nodes: Vec<GraphNode>,

    dist_fn: fn(&[T], &[T]) -> f32,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct NodeWithDist<'a> {
    distance: f32,
    node: &'a GraphNode,
}

impl Eq for NodeWithDist<'_> {}

impl PartialOrd for NodeWithDist<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for NodeWithDist<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.partial_cmp(&other.distance).unwrap()
    }
}

impl<'a> NodeWithDist<'a> {
    pub fn new(node: &'a GraphNode, distance: f32) -> Self {
        Self { node, distance }
    }
}

impl<T: Float, S: VectorStorage<T>> HNSW<T, S> {
    /// Neightbors of a node at a given level.
    fn neighbors(&self, id: u32, level: u16) -> Option<&[u32]> {
        self.nodes
            .get(id as usize)
            .map(|n| n.neighbors[level as usize].as_slice())
    }

    fn node(&self, id: u32) -> Option<&GraphNode> {
        self.nodes.get(id as usize)
    }

    fn distance_to(&self, vector: &[T], idx: u32) -> f32 {
        (self.dist_fn)(vector, self.vectors.get(idx).unwrap())
    }

    /// Search one level of the HNSW graph.
    ///
    /// Parameters
    /// ----------
    /// query : &[T]
    ///     Query vector
    /// ep: &HNSWVector
    ///     Enter point of the search
    /// ef: usize
    ///     The number of neighbors to return.
    /// layer: u16
    ///     The layer to search.
    ///
    /// Returns
    /// -------
    /// Up to `ef` number of neighbors, sorted by their distances to the query vector.
    ///
    fn search_layer<'a>(
        &self,
        query: &[T],
        ep: &'a GraphNode,
        ef: usize,
        layer: u16,
    ) -> BTreeSet<NodeWithDist> {
        let mut visited = HashSet::new();
        let mut candidates = BTreeSet::<NodeWithDist>::new();
        let mut results = BTreeSet::<NodeWithDist>::new();
        visited.insert(ep.id);

        let d = self.distance_to(query, ep.id);
        candidates.insert(NodeWithDist::new(ep, d));

        while !candidates.is_empty() {
            let c = candidates.pop_first().unwrap();
            let furthest = results
                .last()
                .map(|n| n.distance)
                .expect("Result set is empty");
            visited.insert(c.node.id);

            if c.distance > furthest {
                // All elements in result set are evaluated
                break;
            }
            // Unvisited neighbors
            let neighbors = self.neighbors(c.node.id, layer).unwrap();

            for n in neighbors {
                if visited.contains(n) {
                    continue;
                }
                visited.insert(*n);
                let distance = self.distance_to(query, *n);

                let furthest = results
                    .last()
                    .map(|n| n.distance)
                    .expect("Result set is empty");

                if distance < furthest {
                    let new_node =
                        NodeWithDist::new(self.node(*n).expect("Node not found"), distance);

                    results.insert(new_node);
                    candidates.insert(new_node);
                    if results.len() > ef {
                        results.pop_last();
                    }
                }
            }
        }
        results
    }
}

#[cfg(test)]
mod tests {}
