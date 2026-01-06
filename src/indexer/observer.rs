use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use eyre::Result;
use text_chunking::{ChunkError, CodeParseInfo, CodeParseObserver};
use tracing::warn;

use crate::db::GraphData;
use crate::graph::extract_graph_from_tree;

pub(crate) struct ObservedGraph {
    pub(crate) language: String,
    pub(crate) graph: GraphData,
}

pub(crate) struct GraphObserver {
    graphs: Arc<Mutex<HashMap<String, ObservedGraph>>>,
}

impl GraphObserver {
    pub(crate) fn new(graphs: Arc<Mutex<HashMap<String, ObservedGraph>>>) -> Self {
        Self { graphs }
    }
}

impl CodeParseObserver for GraphObserver {
    fn on_parse(&self, info: CodeParseInfo) -> Result<(), ChunkError> {
        let language = info.language_id.clone();

        match extract_graph_from_tree(
            &language,
            info.language,
            info.tree.as_ref(),
            info.source.as_ref(),
        ) {
            Ok(Some(graph)) => {
                let mut guard = self.graphs.lock().expect("graph observer lock poisoned");
                guard.insert(info.file_path.clone(), ObservedGraph { language, graph });
            }
            Ok(None) => {}
            Err(err) => {
                warn!("graph extraction failed for {}: {}", info.file_path, err);
            }
        }

        Ok(())
    }
}
