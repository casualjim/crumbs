pub mod output;
pub mod pipeline;

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::Path;

use crate::config::AppConfig;
use crate::embedding::EmbeddingProvider;
use crate::repository::Repository;

#[derive(Clone, Debug, Default)]
pub struct SelectionOptions {
    pub scope_paths: Vec<String>,
    pub explicit_includes: Vec<String>,
    pub explicit_excludes: Vec<String>,
    pub pinned_items: Vec<String>,
}

/// Shared context available to all stages in the assembly pipeline.
pub struct AssemblyContext<'a> {
    #[allow(dead_code)]
    pub repo_path: &'a Path,
    pub db: &'a dyn Repository,
    pub embedder: Option<&'a dyn EmbeddingProvider>,
    pub reranker: &'a dyn crate::reranker::RerankingProvider,
    #[allow(dead_code)]
    pub config: &'a AppConfig,
    pub selection: SelectionOptions,
}

/// Typed handle into the arena.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Handle<T> {
    index: usize,
    _marker: PhantomData<T>,
}

impl<T> Handle<T> {
    #[allow(dead_code)]
    pub fn index(self) -> usize {
        self.index
    }
}

/// Simple type-indexed arena for sharing data between stages without cloning.
pub struct Arena {
    store: HashMap<TypeId, Vec<Box<dyn Any + Send + Sync>>>,
}

impl Arena {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }

    pub fn insert<T: Send + Sync + 'static>(&mut self, value: T) -> Handle<T> {
        let entry = self.store.entry(TypeId::of::<T>()).or_default();
        entry.push(Box::new(value));
        Handle {
            index: entry.len() - 1,
            _marker: PhantomData,
        }
    }

    pub fn get<T: Send + Sync + 'static>(&self, handle: Handle<T>) -> &T {
        let entry = self
            .store
            .get(&TypeId::of::<T>())
            .expect("missing arena type for handle");
        entry[handle.index]
            .downcast_ref::<T>()
            .expect("arena handle type mismatch")
    }

    #[allow(dead_code)]
    pub fn get_mut<T: Send + Sync + 'static>(&mut self, handle: Handle<T>) -> &mut T {
        let entry = self
            .store
            .get_mut(&TypeId::of::<T>())
            .expect("missing arena type for handle");
        entry[handle.index]
            .downcast_mut::<T>()
            .expect("arena handle type mismatch")
    }
}
