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

use lance_arrow::ArrowFloatType;
use num_traits::Float;

use lance_linalg::MatrixView;

/// Traits for storage implementation to back vectors.
pub trait VectorStorage<T: Float> {
    // TODO: might need to be async if it is backed by disk.
    fn get(&self, idx: u32) -> Option<&[T]>;

    /// Returns the number of vectors in the storage.
    fn len(&self) -> usize;
}

/// A VectorStore backed by in-memory matrix.
pub struct InMemoryVectorStorage<T: ArrowFloatType> {
    data: MatrixView<T>,
}

impl<T: ArrowFloatType> InMemoryVectorStorage<T> {
    pub fn new(data: MatrixView<T>) -> Self {
        Self { data }
    }
}

impl<T: ArrowFloatType> VectorStorage<T::Native> for InMemoryVectorStorage<T> {
    fn get(&self, idx: u32) -> Option<&[T::Native]> {
        self.data.row(idx as usize)
    }

    fn len(&self) -> usize {
        self.data.num_rows()
    }
}
