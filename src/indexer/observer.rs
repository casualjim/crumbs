use std::sync::Arc;

use dashmap::DashMap;
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
    graphs: Arc<DashMap<String, ObservedGraph>>,
}

impl GraphObserver {
    pub(crate) fn new(graphs: Arc<DashMap<String, ObservedGraph>>) -> Self {
        Self { graphs }
    }
}

impl CodeParseObserver for GraphObserver {
    fn on_parse(&self, info: CodeParseInfo) -> Result<(), ChunkError> {
        let language = info.language_id.clone();

        match extract_graph_from_tree(
            &info.file_path,
            &language,
            info.language,
            info.tree.as_ref(),
            info.source.as_ref(),
        ) {
            Ok(Some(graph)) => {
                self.graphs
                    .insert(info.file_path.clone(), ObservedGraph { language, graph });
            }
            Ok(None) => {}
            Err(err) => {
                warn!("graph extraction failed for {}: {}", info.file_path, err);
            }
        }

        Ok(())
    }
}
