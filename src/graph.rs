use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::path::PathBuf;

use eyre::{Result, eyre};
use futures::StreamExt;
use syntastica_queries::{
  GO_LOCALS_CRATES_IO,
  JAVASCRIPT_LOCALS_CRATES_IO,
  PYTHON_LOCALS_CRATES_IO,
  RUST_LOCALS_CRATES_IO,
  TYPESCRIPT_LOCALS_CRATES_IO,
};
use text_chunking::{Chunk, WalkOptions, walk_project, Tokenizer};
use text_chunking::languages::{PeekableReader, detect, get_language};
use tree_sitter::{Parser, Query, QueryCursor, StreamingIterator};
use uuid::Uuid;

use crate::db::{Db, ReferenceRecord, SymbolRecord};

pub struct GraphConfig {
  pub repo_path: PathBuf,
  pub max_chunk_size: usize,
  pub overlap_percentage: f32,
  pub tokenizer: Tokenizer,
  pub max_parallel: usize,
  pub max_file_size: Option<u64>,
  pub large_file_threads: usize,
}

pub struct GraphIndexer {
  db: Db,
  config: GraphConfig,
}

impl GraphIndexer {
  pub fn new(db: Db, config: GraphConfig) -> Self {
    Self { db, config }
  }

  pub async fn index(self) -> Result<()> {
    let existing_hashes = self.db.load_existing_hashes()?;
    let options = WalkOptions {
      max_chunk_size: self.config.max_chunk_size,
      tokenizer: self.config.tokenizer,
      overlap_percentage: self.config.overlap_percentage,
      max_parallel: self.config.max_parallel,
      max_file_size: self.config.max_file_size,
      large_file_threads: self.config.large_file_threads,
      existing_hashes,
      cancel_token: None,
    };

    let mut stream = walk_project(&self.config.repo_path, options);
    while let Some(item) = stream.next().await {
      let project_chunk = item?;
      match project_chunk.chunk {
        Chunk::Delete { file_path } => {
          self.db.delete_file(&file_path)?;
        }
        Chunk::EndOfFile {
          file_path,
          content,
          content_hash,
          ..
        } => {
          let Some(source) = content else {
            return Err(eyre!("missing content for {}", file_path));
          };
          let Some(hash) = content_hash else {
            return Err(eyre!("missing content hash for {}", file_path));
          };

          self.db.clear_file_graph(&file_path)?;
          self.db.upsert_file(&file_path, project_chunk.file_size, hash)?;

          if let Some(language) = detect_language(&file_path, &source).await? {
            if let Some(graph) = extract_graph(&language, &source)? {
              persist_graph(&self.db, &file_path, &language, graph)?;
            }
          }
        }
        _ => {}
      }
    }

    Ok(())
  }
}

struct GraphExtraction {
  symbols: Vec<SymbolRecord>,
  references: Vec<ReferenceRecord>,
  resolutions: Vec<(String, String)>,
}

fn persist_graph(db: &Db, file_path: &str, language: &str, graph: GraphExtraction) -> Result<()> {
  let mut symbols_by_name: HashMap<String, Vec<String>> = HashMap::new();

  for mut symbol in graph.symbols {
    symbol.file_path = file_path.to_string();
    symbol.language = language.to_string();
    symbols_by_name
      .entry(symbol.name.clone())
      .or_default()
      .push(symbol.id.clone());
    db.insert_symbol(&symbol)?;
  }

  for mut reference in graph.references {
    reference.file_path = file_path.to_string();
    reference.language = language.to_string();
    db.insert_reference(&reference)?;

    if let Some(symbol_ids) = symbols_by_name.get(&reference.name) {
      if let Some(symbol_id) = symbol_ids.first() {
        db.link_reference_symbol(&reference.id, symbol_id)?;
      }
    }
  }

  for (reference_id, symbol_id) in graph.resolutions {
    db.link_reference_symbol(&reference_id, &symbol_id)?;
  }

  Ok(())
}

fn extract_graph(language: &str, source: &str) -> Result<Option<GraphExtraction>> {
  let query_source = match language {
    "rust" => RUST_LOCALS_CRATES_IO,
    "python" => PYTHON_LOCALS_CRATES_IO,
    "go" => GO_LOCALS_CRATES_IO,
    "javascript" => JAVASCRIPT_LOCALS_CRATES_IO,
    "typescript" => TYPESCRIPT_LOCALS_CRATES_IO,
    _ => return Ok(None),
  };

  if query_source.trim().is_empty() {
    return Ok(None);
  }

  let language_fn = match get_language(language) {
    Some(lang) => lang,
    None => return Ok(None),
  };
  let ts_language: tree_sitter::Language = language_fn.into();

  let mut parser = Parser::new();
  parser.set_language(&ts_language)?;
  let tree = parser
    .parse(source, None)
    .ok_or_else(|| eyre!("failed to parse source for {}", language))?;

  let query = Query::new(&ts_language, query_source)?;
  let capture_names = query.capture_names();
  let mut cursor = QueryCursor::new();

  let mut symbols = Vec::new();
  let mut references = Vec::new();
  let mut seen = HashSet::new();

  let mut matches = cursor.matches(&query, tree.root_node(), source.as_bytes());
  while let Some(m) = matches.next() {
    for capture in m.captures {
      let name = capture_names
        .get(capture.index as usize)
        .map(|name| name.as_ref())
        .unwrap_or("");
      if !(name.starts_with("local.definition") || name.starts_with("local.reference")) {
        continue;
      }

      let node = capture.node;
      let start_byte = node.start_byte();
      let end_byte = node.end_byte();
      let key = (start_byte, end_byte, name.to_string());
      if !seen.insert(key) {
        continue;
      }

      let text = source
        .get(start_byte..end_byte)
        .unwrap_or("")
        .trim()
        .to_string();
      if text.is_empty() {
        continue;
      }

      if name.starts_with("local.definition") {
        symbols.push(SymbolRecord {
          id: Uuid::now_v7().to_string(),
          file_path: String::new(),
          name: text,
          kind: "definition".to_string(),
          start_byte,
          end_byte,
          language: String::new(),
        });
      } else {
        references.push(ReferenceRecord {
          id: Uuid::now_v7().to_string(),
          file_path: String::new(),
          name: text,
          start_byte,
          end_byte,
          language: String::new(),
        });
      }
    }
  }

  Ok(Some(GraphExtraction {
    symbols,
    references,
    resolutions: Vec::new(),
  }))
}

async fn detect_language(file_path: &str, source: &str) -> Result<Option<String>> {
  let cursor = Cursor::new(source.as_bytes().to_vec());
  let peekable = PeekableReader::new(cursor, 51200);
  let (detection, _reader) = detect(std::path::Path::new(file_path), peekable)
    .await
    .map_err(|(err, _reader)| err)?;

  let Some(detection) = detection else {
    return Ok(None);
  };

  let raw = detection.language().to_ascii_lowercase();
  let normalized = match raw.as_str() {
    "typescriptreact" | "tsx" => "typescript".to_string(),
    "javascriptreact" | "jsx" => "javascript".to_string(),
    other => other.to_string(),
  };
  Ok(Some(normalized))
}
