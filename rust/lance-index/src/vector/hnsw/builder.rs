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

use arrow_array::types::Float32Type;

use crate::vector::graph::InMemoryVectorStorage;

use super::HNSW;

/// HNSW Builder
pub struct HNSWBUilder {
    m_l: f32,
    l_max: u8,

    /// max number of connections ifor each element per layers.
    m_max: usize,
    ef_construction: usize,
}

impl HNSWBUilder {
    /// Build a HNSW graph.
    pub fn build() -> HNSW<f32, InMemoryVectorStorage<Float32Type>> {
        unimplemented!()
    }

    /// Assign random level to a new node
    fn random_level(&self) -> u16 {
        unimplemented!()
    }
}
