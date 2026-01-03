use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use duckdb::types::Value;
use duckdb::{Connection, OptionalExt, params, params_from_iter};
use eyre::{Result, eyre};

pub struct Db {
  conn: Connection,
  embedding_dim: usize,
  vss_loaded: bool,
  duckpgq_loaded: bool,
}

impl Db {
  pub fn open(path: &Path, embedding_dim: Option<usize>) -> Result<Self> {
    let conn = Connection::open(path)?;
    let mut db = Self {
      conn,
      embedding_dim: 0,
      vss_loaded: false,
      duckpgq_loaded: false,
    };
    db.init(embedding_dim)?;
    Ok(db)
  }

  fn init(&mut self, embedding_dim: Option<usize>) -> Result<()> {
    self.conn.execute_batch(
      "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);",
    )?;

    let existing_dim: Option<String> = self
      .conn
      .query_row(
        "SELECT value FROM meta WHERE key = 'embedding_dim'",
        [],
        |row| row.get(0),
      )
      .optional()?;

    let resolved_dim = match (existing_dim, embedding_dim) {
      (Some(value), Some(requested)) => {
        let parsed: usize = value.parse::<usize>().map_err(|err| eyre!(err))?;
        if parsed != requested {
          return Err(eyre!(
            "embedding dimension mismatch: db has {}, requested {}",
            parsed,
            requested
          ));
        }
        parsed
      }
      (Some(value), None) => value.parse::<usize>().map_err(|err| eyre!(err))?,
      (None, Some(requested)) => {
        self.conn.execute(
          "INSERT INTO meta (key, value) VALUES ('embedding_dim', ?)",
          params![requested.to_string()],
        )?;
        requested
      }
      (None, None) => {
        return Err(eyre!(
          "embedding dimension not set; pass --embedding-dim or EMBEDDING_DIM"
        ));
      }
    };

    self.embedding_dim = resolved_dim;

    self.duckpgq_loaded = load_required_extension(&self.conn, "duckpgq", Some("community"))?;
    self.vss_loaded = load_required_extension(&self.conn, "vss", None)?;

    self.create_schema()?;
    Ok(())
  }

  fn create_schema(&mut self) -> Result<()> {
    let create_sql = format!(
      "CREATE TABLE IF NOT EXISTS files (
          path TEXT PRIMARY KEY,
          content_hash BLOB,
          size BIGINT,
          updated_at TIMESTAMP DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS chunks (
          id TEXT PRIMARY KEY,
          file_path TEXT NOT NULL,
          start_byte BIGINT NOT NULL,
          end_byte BIGINT NOT NULL,
          text TEXT NOT NULL,
          kind TEXT NOT NULL,
          ordinal INTEGER NOT NULL,
          embedding FLOAT[{dim}] NOT NULL
        );
        CREATE TABLE IF NOT EXISTS file_chunk_edges (
          file_path TEXT NOT NULL,
          chunk_id TEXT NOT NULL,
          ordinal INTEGER NOT NULL,
          PRIMARY KEY (file_path, chunk_id)
        );
        CREATE TABLE IF NOT EXISTS symbols (
          id TEXT PRIMARY KEY,
          file_path TEXT NOT NULL,
          name TEXT NOT NULL,
          kind TEXT NOT NULL,
          start_byte BIGINT NOT NULL,
          end_byte BIGINT NOT NULL,
          language TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS symbol_references (
          id TEXT PRIMARY KEY,
          file_path TEXT NOT NULL,
          name TEXT NOT NULL,
          start_byte BIGINT NOT NULL,
          end_byte BIGINT NOT NULL,
          language TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS file_symbol_edges (
          file_path TEXT NOT NULL,
          symbol_id TEXT NOT NULL,
          PRIMARY KEY (file_path, symbol_id)
        );
        CREATE TABLE IF NOT EXISTS file_reference_edges (
          file_path TEXT NOT NULL,
          reference_id TEXT NOT NULL,
          PRIMARY KEY (file_path, reference_id)
        );
        CREATE TABLE IF NOT EXISTS reference_symbol_edges (
          reference_id TEXT NOT NULL,
          symbol_id TEXT NOT NULL,
          PRIMARY KEY (reference_id, symbol_id)
        );
        CREATE INDEX IF NOT EXISTS chunks_file_path_idx ON chunks (file_path);
        CREATE INDEX IF NOT EXISTS file_chunk_edges_file_idx ON file_chunk_edges (file_path);
        CREATE INDEX IF NOT EXISTS symbols_file_path_idx ON symbols (file_path);
        CREATE INDEX IF NOT EXISTS references_file_path_idx ON symbol_references (file_path);",
      dim = self.embedding_dim
    );

    self.conn.execute_batch(&create_sql)?;

    if self.vss_loaded {
      let _ = self
        .conn
        .execute_batch("SET hnsw_enable_experimental_persistence=true;");
      let _ = self.conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw \
         ON chunks USING HNSW (embedding) WITH (metric='cosine');",
      );
    }

    if self.duckpgq_loaded {
      let graph_sql = "CREATE OR REPLACE PROPERTY GRAPH code_graph
        VERTEX TABLES (files, chunks, symbols, symbol_references)
        EDGE TABLES (
          file_chunk_edges
            SOURCE KEY (file_path) REFERENCES files (path)
            DESTINATION KEY (chunk_id) REFERENCES chunks (id)
            LABEL contains,
          file_symbol_edges
            SOURCE KEY (file_path) REFERENCES files (path)
            DESTINATION KEY (symbol_id) REFERENCES symbols (id)
            LABEL defines,
          file_reference_edges
            SOURCE KEY (file_path) REFERENCES files (path)
            DESTINATION KEY (reference_id) REFERENCES symbol_references (id)
            LABEL references,
          reference_symbol_edges
            SOURCE KEY (reference_id) REFERENCES symbol_references (id)
            DESTINATION KEY (symbol_id) REFERENCES symbols (id)
            LABEL resolves
        );";
      let _ = self.conn.execute_batch(graph_sql);
    }

    Ok(())
  }

  pub fn load_existing_hashes(&self) -> Result<BTreeMap<PathBuf, [u8; 32]>> {
    let mut stmt = self.conn.prepare(
      "SELECT path, content_hash FROM files WHERE content_hash IS NOT NULL",
    )?;
    let mut rows = stmt.query([])?;
    let mut hashes = BTreeMap::new();
    while let Some(row) = rows.next()? {
      let path: String = row.get(0)?;
      let hash: Vec<u8> = row.get(1)?;
      if hash.len() != 32 {
        continue;
      }
      let mut buf = [0u8; 32];
      buf.copy_from_slice(&hash);
      hashes.insert(PathBuf::from(path), buf);
    }
    Ok(hashes)
  }

  pub fn clear_file_chunks(&self, file_path: &str) -> Result<()> {
    self
      .conn
      .execute("DELETE FROM file_chunk_edges WHERE file_path = ?", params![file_path])?;
    self
      .conn
      .execute("DELETE FROM chunks WHERE file_path = ?", params![file_path])?;
    Ok(())
  }

  pub fn clear_file_graph(&self, file_path: &str) -> Result<()> {
    self.conn.execute(
      "DELETE FROM reference_symbol_edges WHERE symbol_id IN \
       (SELECT id FROM symbols WHERE file_path = ?)",
      params![file_path],
    )?;
    self.conn.execute(
      "DELETE FROM reference_symbol_edges WHERE reference_id IN \
       (SELECT id FROM symbol_references WHERE file_path = ?)",
      params![file_path],
    )?;
    self
      .conn
      .execute("DELETE FROM file_reference_edges WHERE file_path = ?", params![file_path])?;
    self
      .conn
      .execute("DELETE FROM symbol_references WHERE file_path = ?", params![file_path])?;
    self
      .conn
      .execute("DELETE FROM file_symbol_edges WHERE file_path = ?", params![file_path])?;
    self
      .conn
      .execute("DELETE FROM symbols WHERE file_path = ?", params![file_path])?;
    Ok(())
  }

  pub fn delete_file(&self, file_path: &str) -> Result<()> {
    self.clear_file_graph(file_path)?;
    self.clear_file_chunks(file_path)?;
    self
      .conn
      .execute("DELETE FROM files WHERE path = ?", params![file_path])?;
    Ok(())
  }

  pub fn upsert_file(&self, file_path: &str, file_size: u64, content_hash: [u8; 32]) -> Result<()> {
    self.conn.execute(
      "INSERT INTO files (path, content_hash, size, updated_at)
       VALUES (?, ?, ?, now())
       ON CONFLICT(path) DO UPDATE SET
         content_hash = excluded.content_hash,
         size = excluded.size,
         updated_at = now()",
      params![file_path, content_hash.to_vec(), file_size as i64],
    )?;
    Ok(())
  }

  pub fn ensure_file(&self, file_path: &str, file_size: u64) -> Result<()> {
    self.conn.execute(
      "INSERT INTO files (path, size, updated_at)
       VALUES (?, ?, now())
       ON CONFLICT(path) DO UPDATE SET
         size = excluded.size,
         updated_at = now()",
      params![file_path, file_size as i64],
    )?;
    Ok(())
  }

  pub fn insert_chunk(
    &self,
    record: &ChunkRecord,
    embedding: &[f32],
  ) -> Result<()> {
    self.ensure_file_exists(&record.file_path)?;
    if record.start_byte > record.end_byte {
      return Err(eyre!(
        "chunk byte range invalid: start {} > end {}",
        record.start_byte,
        record.end_byte
      ));
    }
    if embedding.len() != self.embedding_dim {
      return Err(eyre!(
        "embedding dimension mismatch: expected {}, got {}",
        self.embedding_dim,
        embedding.len()
      ));
    }

    let array_value_sql = array_value_placeholder(self.embedding_dim);
    let sql = format!(
      "INSERT INTO chunks \
      (id, file_path, start_byte, end_byte, text, kind, ordinal, embedding) \
      VALUES (?, ?, ?, ?, ?, ?, ?, {array}::FLOAT[{dim}])",
      array = array_value_sql,
      dim = self.embedding_dim
    );

    let mut params_vec = Vec::with_capacity(7 + embedding.len());
    params_vec.push(Value::Text(record.id.clone()));
    params_vec.push(Value::Text(record.file_path.clone()));
    params_vec.push(Value::BigInt(record.start_byte as i64));
    params_vec.push(Value::BigInt(record.end_byte as i64));
    params_vec.push(Value::Text(record.text.clone()));
    params_vec.push(Value::Text(record.kind.clone()));
    params_vec.push(Value::Int(record.ordinal as i32));
    params_vec.extend(embedding.iter().copied().map(Value::Float));

    let mut stmt = self.conn.prepare(&sql)?;
    stmt.execute(params_from_iter(params_vec))?;

    self.conn.execute(
      "INSERT INTO file_chunk_edges (file_path, chunk_id, ordinal) VALUES (?, ?, ?)",
      params![record.file_path, record.id, record.ordinal as i32],
    )?;

    Ok(())
  }

  pub fn insert_symbol(&self, record: &SymbolRecord) -> Result<()> {
    self.ensure_file_exists(&record.file_path)?;
    self.conn.execute(
      "INSERT INTO symbols (id, file_path, name, kind, start_byte, end_byte, language) \
       VALUES (?, ?, ?, ?, ?, ?, ?)",
      params![
        record.id,
        record.file_path,
        record.name,
        record.kind,
        record.start_byte as i64,
        record.end_byte as i64,
        record.language
      ],
    )?;
    self.conn.execute(
      "INSERT INTO file_symbol_edges (file_path, symbol_id) VALUES (?, ?)",
      params![record.file_path, record.id],
    )?;
    Ok(())
  }

  pub fn insert_reference(&self, record: &ReferenceRecord) -> Result<()> {
    self.ensure_file_exists(&record.file_path)?;
    self.conn.execute(
      "INSERT INTO symbol_references (id, file_path, name, start_byte, end_byte, language) \
       VALUES (?, ?, ?, ?, ?, ?)",
      params![
        record.id,
        record.file_path,
        record.name,
        record.start_byte as i64,
        record.end_byte as i64,
        record.language
      ],
    )?;
    self.conn.execute(
      "INSERT INTO file_reference_edges (file_path, reference_id) VALUES (?, ?)",
      params![record.file_path, record.id],
    )?;
    Ok(())
  }

  pub fn link_reference_symbol(&self, reference_id: &str, symbol_id: &str) -> Result<()> {
    self.ensure_reference_exists(reference_id)?;
    self.ensure_symbol_exists(symbol_id)?;
    self.conn.execute(
      "INSERT INTO reference_symbol_edges (reference_id, symbol_id) VALUES (?, ?)",
      params![reference_id, symbol_id],
    )?;
    Ok(())
  }

  pub fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<SearchRow>> {
    if query_embedding.len() != self.embedding_dim {
      return Err(eyre!(
        "embedding dimension mismatch: expected {}, got {}",
        self.embedding_dim,
        query_embedding.len()
      ));
    }

    let array_value_sql = array_value_placeholder(self.embedding_dim);
    let sql = format!(
      "SELECT file_path, start_byte, end_byte, text, \
       array_cosine_distance(embedding, {array}::FLOAT[{dim}]) AS distance \
       FROM chunks ORDER BY distance ASC LIMIT ?",
      array = array_value_sql,
      dim = self.embedding_dim
    );

    let mut params_vec = Vec::with_capacity(query_embedding.len() + 1);
    params_vec.extend(query_embedding.iter().copied().map(Value::Float));
    params_vec.push(Value::BigInt(limit as i64));

    let mut stmt = self.conn.prepare(&sql)?;
    let mut rows = stmt.query(params_from_iter(params_vec))?;
    let mut results = Vec::new();
    while let Some(row) = rows.next()? {
      results.push(SearchRow {
        file_path: row.get(0)?,
      start_byte: row.get(1)?,
      end_byte: row.get(2)?,
        text: row.get(3)?,
        distance: row.get(4)?,
      });
    }
    Ok(results)
  }

  fn ensure_file_exists(&self, file_path: &str) -> Result<()> {
    let exists: Option<i64> = self
      .conn
      .query_row(
        "SELECT 1 FROM files WHERE path = ?",
        params![file_path],
        |row| row.get(0),
      )
      .optional()?;
    if exists.is_none() {
      return Err(eyre!("file not found: {}", file_path));
    }
    Ok(())
  }

  fn ensure_reference_exists(&self, reference_id: &str) -> Result<()> {
    let exists: Option<i64> = self
      .conn
      .query_row(
        "SELECT 1 FROM symbol_references WHERE id = ?",
        params![reference_id],
        |row| row.get(0),
      )
      .optional()?;
    if exists.is_none() {
      return Err(eyre!("reference not found: {}", reference_id));
    }
    Ok(())
  }

  fn ensure_symbol_exists(&self, symbol_id: &str) -> Result<()> {
    let exists: Option<i64> = self
      .conn
      .query_row(
        "SELECT 1 FROM symbols WHERE id = ?",
        params![symbol_id],
        |row| row.get(0),
      )
      .optional()?;
    if exists.is_none() {
      return Err(eyre!("symbol not found: {}", symbol_id));
    }
    Ok(())
  }
}

pub struct ChunkRecord {
  pub id: String,
  pub file_path: String,
  pub start_byte: usize,
  pub end_byte: usize,
  pub text: String,
  pub kind: String,
  pub ordinal: usize,
}

pub struct SymbolRecord {
  pub id: String,
  pub file_path: String,
  pub name: String,
  pub kind: String,
  pub start_byte: usize,
  pub end_byte: usize,
  pub language: String,
}

pub struct ReferenceRecord {
  pub id: String,
  pub file_path: String,
  pub name: String,
  pub start_byte: usize,
  pub end_byte: usize,
  pub language: String,
}

pub struct SearchRow {
  pub file_path: String,
  pub start_byte: i64,
  pub end_byte: i64,
  pub text: String,
  pub distance: f64,
}

fn array_value_placeholder(count: usize) -> String {
  let mut vars = "?,"
    .repeat(count)
    .trim_end_matches(',')
    .to_string();
  if vars.is_empty() {
    vars.push('?');
  }
  format!("array_value({vars})")
}

fn load_required_extension(conn: &Connection, name: &str, from: Option<&str>) -> Result<bool> {
  if conn.execute_batch(&format!("LOAD {name};")).is_ok() {
    return Ok(true);
  }
  let install = match from {
    Some(source) => format!("INSTALL {name} FROM {source}; LOAD {name};"),
    None => format!("INSTALL {name}; LOAD {name};"),
  };
  conn.execute_batch(&install)?;
  Ok(true)
}

#[cfg(test)]
mod tests {
  use super::*;
  use duckdb::Connection;
  use tempfile::TempDir;

  #[test]
  fn db_enforces_embedding_dim() -> Result<()> {
    let dir = TempDir::new()?;
    let db_path = dir.path().join("context.duckdb");

    let _db = Db::open(&db_path, Some(2))?;
    let mismatch = Db::open(&db_path, Some(3));
    assert!(mismatch.is_err(), "expected embedding_dim mismatch error");
    Ok(())
  }

  #[test]
  fn db_rejects_chunk_without_file_row() -> Result<()> {
    let dir = TempDir::new()?;
    let db_path = dir.path().join("context.duckdb");
    let db = Db::open(&db_path, Some(2))?;

    let record = ChunkRecord {
      id: "chunk-1".to_string(),
      file_path: "missing.rs".to_string(),
      start_byte: 0,
      end_byte: 5,
      text: "hello".to_string(),
      kind: "text".to_string(),
      ordinal: 0,
    };
    let result = db.insert_chunk(&record, &[0.1, 0.2]);

    assert!(
      result.is_err(),
      "expected insert_chunk to fail when file row is missing"
    );
    Ok(())
  }

  #[test]
  fn db_rejects_chunk_with_inverted_bounds() -> Result<()> {
    let dir = TempDir::new()?;
    let db_path = dir.path().join("context.duckdb");
    let db = Db::open(&db_path, Some(2))?;
    let hash = [0u8; 32];

    db.upsert_file("existing.rs", 12, hash)?;

    let record = ChunkRecord {
      id: "chunk-2".to_string(),
      file_path: "existing.rs".to_string(),
      start_byte: 10,
      end_byte: 5,
      text: "broken".to_string(),
      kind: "text".to_string(),
      ordinal: 0,
    };
    let result = db.insert_chunk(&record, &[0.1, 0.2]);

    assert!(
      result.is_err(),
      "expected insert_chunk to reject inverted byte ranges"
    );
    Ok(())
  }

  #[test]
  fn db_rejects_symbol_without_file_row() -> Result<()> {
    let dir = TempDir::new()?;
    let db_path = dir.path().join("context.duckdb");
    let db = Db::open(&db_path, Some(2))?;

    let record = SymbolRecord {
      id: "symbol-1".to_string(),
      file_path: "missing.rs".to_string(),
      name: "add".to_string(),
      kind: "definition".to_string(),
      start_byte: 0,
      end_byte: 3,
      language: "rust".to_string(),
    };
    let result = db.insert_symbol(&record);

    assert!(
      result.is_err(),
      "expected insert_symbol to fail when file row is missing"
    );
    Ok(())
  }

  #[test]
  fn db_rejects_reference_without_file_row() -> Result<()> {
    let dir = TempDir::new()?;
    let db_path = dir.path().join("context.duckdb");
    let db = Db::open(&db_path, Some(2))?;

    let record = ReferenceRecord {
      id: "ref-1".to_string(),
      file_path: "missing.rs".to_string(),
      name: "add".to_string(),
      start_byte: 10,
      end_byte: 13,
      language: "rust".to_string(),
    };
    let result = db.insert_reference(&record);

    assert!(
      result.is_err(),
      "expected insert_reference to fail when file row is missing"
    );
    Ok(())
  }

  #[test]
  fn db_rejects_reference_symbol_links_for_missing_ids() -> Result<()> {
    let dir = TempDir::new()?;
    let db_path = dir.path().join("context.duckdb");
    let db = Db::open(&db_path, Some(2))?;

    let result = db.link_reference_symbol("missing-ref", "missing-sym");

    assert!(
      result.is_err(),
      "expected link_reference_symbol to fail for missing ids"
    );
    Ok(())
  }

  #[test]
  fn db_delete_file_cleans_cross_file_reference_edges() -> Result<()> {
    let dir = TempDir::new()?;
    let db_path = dir.path().join("context.duckdb");
    let db = Db::open(&db_path, Some(2))?;
    let hash = [0u8; 32];

    db.upsert_file("a.rs", 12, hash)?;
    db.upsert_file("b.rs", 12, hash)?;

    let symbol = SymbolRecord {
      id: "sym-1".to_string(),
      file_path: "a.rs".to_string(),
      name: "add".to_string(),
      kind: "definition".to_string(),
      start_byte: 0,
      end_byte: 3,
      language: "rust".to_string(),
    };
    let reference = ReferenceRecord {
      id: "ref-2".to_string(),
      file_path: "b.rs".to_string(),
      name: "add".to_string(),
      start_byte: 10,
      end_byte: 13,
      language: "rust".to_string(),
    };

    db.insert_symbol(&symbol)?;
    db.insert_reference(&reference)?;
    db.link_reference_symbol(&reference.id, &symbol.id)?;

    db.delete_file("a.rs")?;

    let conn = Connection::open(&db_path)?;
    let edge_count: i64 =
      conn.query_row("SELECT COUNT(*) FROM reference_symbol_edges", [], |row| row.get(0))?;

    assert_eq!(
      edge_count, 0,
      "expected reference_symbol_edges to be removed when symbols are deleted"
    );
    Ok(())
  }
}
