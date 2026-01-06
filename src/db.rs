use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use duckdb::types::Value;
use duckdb::{Connection, OptionalExt, params, params_from_iter};
use eyre::{Result, eyre};
use tracing::warn;

use crate::repository::Repository;

pub struct Db {
    conn: Connection,
    embedding_dim: usize,
    vss_loaded: bool,
    duckpgq_loaded: bool,
    fts_loaded: bool,
}

impl Db {
    pub fn open(path: &Path, embedding_dim: Option<usize>) -> Result<Self> {
        let conn = Connection::open(path)?;
        let mut db = Self {
            conn,
            embedding_dim: 0,
            vss_loaded: false,
            duckpgq_loaded: false,
            fts_loaded: false,
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

        self.duckpgq_loaded = load_optional_extension(&self.conn, "duckpgq", Some("community"))?;
        self.vss_loaded = load_optional_extension(&self.conn, "vss", None)?;
        self.fts_loaded = load_optional_extension(&self.conn, "fts", None)?;

        self.create_schema()?;
        Ok(())
    }

    fn create_schema(&mut self) -> Result<()> {
        let create_sql = format!(
            "CREATE TABLE IF NOT EXISTS files (
          path TEXT PRIMARY KEY,
          content_hash BLOB,
          size BIGINT,
          primary_language TEXT,
          updated_at TIMESTAMP DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS file_commit_edges (
          file_path TEXT NOT NULL REFERENCES files(path),
          commit_id TEXT NOT NULL,
          PRIMARY KEY (file_path, commit_id)
        );
        CREATE TABLE IF NOT EXISTS file_cochange_edges (
          src_path TEXT NOT NULL REFERENCES files(path),
          dst_path TEXT NOT NULL REFERENCES files(path),
          commit_count BIGINT NOT NULL,
          weight DOUBLE NOT NULL,
          PRIMARY KEY (src_path, dst_path)
        );
        CREATE TABLE IF NOT EXISTS file_dependency_edges (
          src_path TEXT NOT NULL REFERENCES files(path),
          dst_path TEXT NOT NULL REFERENCES files(path),
          reference_count BIGINT NOT NULL,
          PRIMARY KEY (src_path, dst_path)
        );
        CREATE TABLE IF NOT EXISTS chunks (
          id TEXT PRIMARY KEY,
          file_path TEXT NOT NULL REFERENCES files(path),
          start_byte BIGINT NOT NULL,
          end_byte BIGINT NOT NULL,
          chunk_hash BLOB NOT NULL,
          start_line INTEGER NOT NULL,
          end_line INTEGER NOT NULL,
          text TEXT NOT NULL,
          kind TEXT NOT NULL,
          ordinal INTEGER NOT NULL,
          tokens INTEGER[],
          embedding FLOAT[{dim}] NOT NULL
        );
        CREATE TABLE IF NOT EXISTS file_chunk_edges (
          file_path TEXT NOT NULL REFERENCES files(path),
          chunk_id TEXT NOT NULL REFERENCES chunks(id),
          ordinal INTEGER NOT NULL,
          PRIMARY KEY (file_path, chunk_id)
        );
        CREATE TABLE IF NOT EXISTS symbols (
          id TEXT PRIMARY KEY,
          file_path TEXT NOT NULL REFERENCES files(path),
          name TEXT NOT NULL,
          kind TEXT NOT NULL,
          start_byte BIGINT NOT NULL,
          end_byte BIGINT NOT NULL,
          language TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS symbol_references (
          id TEXT PRIMARY KEY,
          file_path TEXT NOT NULL REFERENCES files(path),
          name TEXT NOT NULL,
          start_byte BIGINT NOT NULL,
          end_byte BIGINT NOT NULL,
          language TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS file_symbol_edges (
          file_path TEXT NOT NULL REFERENCES files(path),
          symbol_id TEXT NOT NULL REFERENCES symbols(id),
          PRIMARY KEY (file_path, symbol_id)
        );
        CREATE TABLE IF NOT EXISTS file_reference_edges (
          file_path TEXT NOT NULL REFERENCES files(path),
          reference_id TEXT NOT NULL REFERENCES symbol_references(id),
          PRIMARY KEY (file_path, reference_id)
        );
        CREATE TABLE IF NOT EXISTS reference_symbol_edges (
          reference_id TEXT NOT NULL REFERENCES symbol_references(id),
          symbol_id TEXT NOT NULL REFERENCES symbols(id),
          PRIMARY KEY (reference_id, symbol_id)
        );
        CREATE INDEX IF NOT EXISTS chunks_file_path_idx ON chunks (file_path);
        CREATE INDEX IF NOT EXISTS file_commit_edges_file_idx ON file_commit_edges (file_path);
        CREATE INDEX IF NOT EXISTS file_commit_edges_commit_idx ON file_commit_edges (commit_id);
        CREATE INDEX IF NOT EXISTS file_cochange_edges_src_idx ON file_cochange_edges (src_path);
        CREATE INDEX IF NOT EXISTS file_cochange_edges_dst_idx ON file_cochange_edges (dst_path);
        CREATE INDEX IF NOT EXISTS file_dependency_edges_src_idx ON file_dependency_edges (src_path);
        CREATE INDEX IF NOT EXISTS file_dependency_edges_dst_idx ON file_dependency_edges (dst_path);
        CREATE INDEX IF NOT EXISTS file_chunk_edges_file_idx ON file_chunk_edges (file_path);
        CREATE INDEX IF NOT EXISTS symbols_file_path_idx ON symbols (file_path);
        CREATE INDEX IF NOT EXISTS references_file_path_idx ON symbol_references (file_path);",
            dim = self.embedding_dim
        );

        self.conn.execute_batch(&create_sql)?;
        self.ensure_chunk_tokens_column()?;

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
            LABEL resolves,
          file_cochange_edges
            SOURCE KEY (src_path) REFERENCES files (path)
            DESTINATION KEY (dst_path) REFERENCES files (path)
            LABEL cochanges,
          file_dependency_edges
            SOURCE KEY (src_path) REFERENCES files (path)
            DESTINATION KEY (dst_path) REFERENCES files (path)
            LABEL depends_on
        );";
            let _ = self.conn.execute_batch(graph_sql);
        }

        Ok(())
    }

    fn ensure_chunk_tokens_column(&self) -> Result<()> {
        if self.column_exists("chunks", "tokens")? {
            return Ok(());
        }
        self.conn
            .execute("ALTER TABLE chunks ADD COLUMN tokens INTEGER[]", [])?;
        Ok(())
    }

    fn column_exists(&self, table: &str, column: &str) -> Result<bool> {
        let pragma = format!("PRAGMA table_info('{table}')");
        let mut stmt = self.conn.prepare(&pragma)?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let name: String = row.get(1)?;
            if name == column {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn clear_file_chunks(&self, file_path: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM file_chunk_edges WHERE file_path = ?",
            params![file_path],
        )?;
        self.conn
            .execute("DELETE FROM chunks WHERE file_path = ?", params![file_path])?;
        Ok(())
    }

    fn clear_file_graph(&self, file_path: &str) -> Result<()> {
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
        self.conn.execute(
            "DELETE FROM file_reference_edges WHERE file_path = ?",
            params![file_path],
        )?;
        self.conn.execute(
            "DELETE FROM symbol_references WHERE file_path = ?",
            params![file_path],
        )?;
        self.conn.execute(
            "DELETE FROM file_symbol_edges WHERE file_path = ?",
            params![file_path],
        )?;
        self.conn.execute(
            "DELETE FROM symbols WHERE file_path = ?",
            params![file_path],
        )?;
        self.conn.execute(
            "DELETE FROM file_dependency_edges WHERE src_path = ? OR dst_path = ?",
            params![file_path, file_path],
        )?;
        Ok(())
    }

    fn clear_file_history(&self, file_path: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM file_commit_edges WHERE file_path = ?",
            params![file_path],
        )?;
        self.conn.execute(
            "DELETE FROM file_cochange_edges WHERE src_path = ? OR dst_path = ?",
            params![file_path, file_path],
        )?;
        Ok(())
    }

    fn upsert_file(
        &self,
        file_path: &str,
        file_size: u64,
        content_hash: [u8; 32],
        primary_language: Option<&str>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO files (path, content_hash, size, primary_language, updated_at)
       VALUES (?, ?, ?, ?, now())
       ON CONFLICT(path) DO UPDATE SET
         content_hash = excluded.content_hash,
         size = excluded.size,
         primary_language = excluded.primary_language,
         updated_at = now()",
            params![
                file_path,
                content_hash.to_vec(),
                file_size as i64,
                primary_language
            ],
        )?;
        Ok(())
    }

    fn ensure_file(
        &self,
        file_path: &str,
        file_size: u64,
        primary_language: Option<&str>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO files (path, size, primary_language, updated_at)
       VALUES (?, ?, ?, now())
       ON CONFLICT(path) DO UPDATE SET
         size = excluded.size,
         primary_language = excluded.primary_language,
         updated_at = now()",
            params![file_path, file_size as i64, primary_language],
        )?;
        Ok(())
    }

    fn insert_chunk(&self, record: &ChunkRecord, embedding: &[f32]) -> Result<()> {
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
        let (tokens_sql, tokens_params_len) = match record
            .tokens
            .as_ref()
            .filter(|tokens| !tokens.is_empty())
        {
            Some(tokens) => (format!("{}::INTEGER[]", array_value_placeholder(tokens.len())), tokens.len()),
            None => ("NULL".to_string(), 0),
        };
        let sql = format!(
            "INSERT INTO chunks \
      (id, file_path, start_byte, end_byte, chunk_hash, start_line, end_line, text, kind, ordinal, tokens, embedding) \
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {tokens_sql}, {array}::FLOAT[{dim}])",
            tokens_sql = tokens_sql,
            array = array_value_sql,
            dim = self.embedding_dim
        );

        let mut params_vec = Vec::with_capacity(10 + tokens_params_len + embedding.len());
        params_vec.push(Value::Text(record.id.clone()));
        params_vec.push(Value::Text(record.file_path.clone()));
        params_vec.push(Value::BigInt(record.start_byte as i64));
        params_vec.push(Value::BigInt(record.end_byte as i64));
        params_vec.push(Value::Blob(record.chunk_hash.to_vec()));
        params_vec.push(Value::BigInt(record.start_line as i64));
        params_vec.push(Value::BigInt(record.end_line as i64));
        params_vec.push(Value::Text(record.text.clone()));
        params_vec.push(Value::Text(record.kind.clone()));
        params_vec.push(Value::Int(record.ordinal as i32));
        if let Some(tokens) = record.tokens.as_ref().filter(|tokens| !tokens.is_empty()) {
            params_vec.extend(tokens.iter().map(|token| Value::Int(*token as i32)));
        }
        params_vec.extend(embedding.iter().copied().map(Value::Float));

        let mut stmt = self.conn.prepare(&sql)?;
        stmt.execute(params_from_iter(params_vec))?;

        self.conn.execute(
            "INSERT INTO file_chunk_edges (file_path, chunk_id, ordinal) VALUES (?, ?, ?)",
            params![record.file_path, record.id, record.ordinal as i32],
        )?;

        Ok(())
    }

    fn insert_symbol(&self, record: &SymbolRecord) -> Result<()> {
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

    fn insert_reference(&self, record: &ReferenceRecord) -> Result<()> {
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

    fn link_reference_symbol(&self, reference_id: &str, symbol_id: &str) -> Result<()> {
        self.ensure_reference_exists(reference_id)?;
        self.ensure_symbol_exists(symbol_id)?;
        self.conn.execute(
            "INSERT INTO reference_symbol_edges (reference_id, symbol_id) VALUES (?, ?)",
            params![reference_id, symbol_id],
        )?;
        Ok(())
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

    fn with_transaction<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Db) -> Result<T>,
    {
        self.conn.execute_batch("BEGIN TRANSACTION")?;
        let result = f(self);
        match result {
            Ok(value) => {
                if let Err(err) = self.conn.execute_batch("COMMIT") {
                    let _ = self.conn.execute_batch("ROLLBACK");
                    return Err(err.into());
                }
                Ok(value)
            }
            Err(err) => {
                let _ = self.conn.execute_batch("ROLLBACK");
                Err(err)
            }
        }
    }

    fn persist_graph(&self, file_path: &str, language: &str, graph: GraphData) -> Result<()> {
        let mut symbols_by_name: HashMap<String, Vec<String>> = HashMap::new();

        for mut symbol in graph.symbols {
            symbol.file_path = file_path.to_string();
            symbol.language = language.to_string();
            symbols_by_name
                .entry(symbol.name.clone())
                .or_default()
                .push(symbol.id.clone());
            self.insert_symbol(&symbol)?;
        }

        for mut reference in graph.references {
            reference.file_path = file_path.to_string();
            reference.language = language.to_string();
            self.insert_reference(&reference)?;

            if let Some(symbol_ids) = symbols_by_name.get(&reference.name)
                && symbol_ids.len() == 1
            {
                let symbol_id = &symbol_ids[0];
                self.link_reference_symbol(&reference.id, symbol_id)?;
            }
        }

        for (reference_id, symbol_id) in graph.resolutions {
            self.link_reference_symbol(&reference_id, &symbol_id)?;
        }

        Ok(())
    }
}

pub struct ChunkRecord {
    pub id: String,
    pub file_path: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub chunk_hash: [u8; 32],
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub kind: String,
    pub ordinal: usize,
    pub tokens: Option<Vec<u32>>,
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

pub struct GraphData {
    pub symbols: Vec<SymbolRecord>,
    pub references: Vec<ReferenceRecord>,
    pub resolutions: Vec<(String, String)>,
}

pub struct SearchRow {
    pub id: String,
    pub file_path: String,
    pub start_byte: i64,
    pub end_byte: i64,
    pub chunk_hash: [u8; 32],
    pub start_line: i64,
    pub end_line: i64,
    pub text: String,
    pub distance: f64,
}

pub struct FtsRow {
    pub id: String,
    pub file_path: String,
    pub start_byte: i64,
    pub end_byte: i64,
    pub chunk_hash: [u8; 32],
    pub start_line: i64,
    pub end_line: i64,
    pub text: String,
    pub score: f64,
}

fn array_value_placeholder(count: usize) -> String {
    let mut vars = "?,".repeat(count).trim_end_matches(',').to_string();
    if vars.is_empty() {
        vars.push('?');
    }
    format!("array_value({vars})")
}

fn load_optional_extension(conn: &Connection, name: &str, from: Option<&str>) -> Result<bool> {
    if conn.execute_batch(&format!("LOAD {name};")).is_ok() {
        return Ok(true);
    }
    let install = match from {
        Some(source) => format!("INSTALL {name} FROM {source}; LOAD {name};"),
        None => format!("INSTALL {name}; LOAD {name};"),
    };
    if let Err(err) = conn.execute_batch(&install) {
        warn!("Failed to load/install extension {name}: {err}");
        return Ok(false);
    }
    Ok(true)
}

impl Repository for Db {
    fn load_existing_hashes(&self) -> Result<BTreeMap<PathBuf, [u8; 32]>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path, content_hash FROM files WHERE content_hash IS NOT NULL")?;
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

    fn delete_file(&self, file_path: &str) -> Result<()> {
        self.clear_file_graph(file_path)?;
        self.clear_file_chunks(file_path)?;
        self.clear_file_history(file_path)?;
        self.conn
            .execute("DELETE FROM files WHERE path = ?", params![file_path])?;
        Ok(())
    }

    fn replace_file_chunks(
        &self,
        file_path: &str,
        file_size: u64,
        content_hash: [u8; 32],
        primary_language: Option<String>,
        chunks: &[ChunkRecord],
        embeddings: &[Vec<f32>],
    ) -> Result<()> {
        if chunks.len() != embeddings.len() {
            return Err(eyre!(
                "chunk/embedding count mismatch: {} chunks vs {} embeddings",
                chunks.len(),
                embeddings.len()
            ));
        }
        for record in chunks {
            if record.file_path != file_path {
                return Err(eyre!(
                    "chunk file_path mismatch: expected {}, got {}",
                    file_path,
                    record.file_path
                ));
            }
        }

        let primary_language = primary_language.as_deref();
        self.with_transaction(|db| {
            db.ensure_file(file_path, file_size, primary_language)?;
            db.clear_file_chunks(file_path)?;
            for (record, embedding) in chunks.iter().zip(embeddings.iter()) {
                db.insert_chunk(record, embedding)?;
            }
            db.upsert_file(file_path, file_size, content_hash, primary_language)?;
            Ok(())
        })?;
        Ok(())
    }

    fn refresh_fts_index(&self) -> Result<()> {
        if !self.fts_loaded {
            warn!("fts extension not available; skipping full-text index refresh");
            return Ok(());
        }
        let _ = self.conn.execute_batch("PRAGMA drop_fts_index('chunks');");
        self.conn
            .execute_batch("PRAGMA create_fts_index('chunks', 'id', 'text');")?;
        Ok(())
    }

    fn replace_file_graph(
        &self,
        file_path: &str,
        file_size: u64,
        content_hash: [u8; 32],
        language: &str,
        primary_language: Option<String>,
        graph: GraphData,
    ) -> Result<()> {
        let primary_language = primary_language.as_deref();
        self.with_transaction(|db| {
            db.ensure_file(file_path, file_size, primary_language)?;
            db.clear_file_graph(file_path)?;
            db.persist_graph(file_path, language, graph)?;
            db.upsert_file(file_path, file_size, content_hash, primary_language)?;
            Ok(())
        })?;
        Ok(())
    }

    fn list_files(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT path FROM files")?;
        let mut rows = stmt.query([])?;
        let mut files = Vec::new();
        while let Some(row) = rows.next()? {
            let path: String = row.get(0)?;
            files.push(path);
        }
        Ok(files)
    }

    fn file_primary_language(&self, file_path: &str) -> Result<Option<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT primary_language FROM files WHERE path = ?")?;
        let mut rows = stmt.query(params![file_path])?;
        if let Some(row) = rows.next()? {
            let lang: Option<String> = row.get(0)?;
            Ok(lang)
        } else {
            Ok(None)
        }
    }

    fn replace_history_edges(
        &self,
        file_commit_edges: &[(String, String)],
        cochange_edges: &[(String, String, i64, f64)],
    ) -> Result<()> {
        self.with_transaction(|db| {
            db.conn.execute("DELETE FROM file_commit_edges", [])?;
            db.conn.execute("DELETE FROM file_cochange_edges", [])?;

            for (file_path, commit_id) in file_commit_edges {
                db.conn.execute(
                    "INSERT INTO file_commit_edges (file_path, commit_id) VALUES (?, ?)",
                    params![file_path, commit_id],
                )?;
            }

            for (src, dst, commit_count, weight) in cochange_edges {
                db.conn.execute(
                    "INSERT INTO file_cochange_edges \
                     (src_path, dst_path, commit_count, weight) VALUES (?, ?, ?, ?)",
                    params![src, dst, commit_count, weight],
                )?;
            }

            Ok(())
        })?;
        Ok(())
    }

    fn vss_loaded(&self) -> bool {
        self.vss_loaded
    }

    fn fts_loaded(&self) -> bool {
        self.fts_loaded
    }

    fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<SearchRow>> {
        if !self.vss_loaded {
            return Err(eyre!(
                "vss extension not available; install/load it to enable vector search"
            ));
        }
        if query_embedding.len() != self.embedding_dim {
            return Err(eyre!(
                "embedding dimension mismatch: expected {}, got {}",
                self.embedding_dim,
                query_embedding.len()
            ));
        }

        let array_value_sql = array_value_placeholder(self.embedding_dim);
        let sql = format!(
            "SELECT id, file_path, start_byte, end_byte, chunk_hash, start_line, end_line, text, \
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
            let chunk_hash: Vec<u8> = row.get(4)?;
            if chunk_hash.len() != 32 {
                return Err(eyre!(
                    "invalid chunk_hash length for {}",
                    row.get::<_, String>(1)?
                ));
            }
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&chunk_hash);
            results.push(SearchRow {
                id: row.get(0)?,
                file_path: row.get(1)?,
                start_byte: row.get(2)?,
                end_byte: row.get(3)?,
                chunk_hash: hash,
                start_line: row.get(5)?,
                end_line: row.get(6)?,
                text: row.get(7)?,
                distance: row.get(8)?,
            });
        }
        Ok(results)
    }

    fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<FtsRow>> {
        if !self.fts_loaded {
            return Err(eyre!(
                "fts extension not available; install/load it to enable full-text search"
            ));
        }

        let sql = "SELECT id, file_path, start_byte, end_byte, chunk_hash, start_line, end_line, text, score \
      FROM ( \
        SELECT id, file_path, start_byte, end_byte, chunk_hash, start_line, end_line, text, \
          fts_main_chunks.match_bm25(id, ?) AS score \
        FROM chunks \
      ) sq \
      WHERE score IS NOT NULL \
      ORDER BY score DESC \
      LIMIT ?";

        let mut stmt = self.conn.prepare(sql)?;
        let mut rows = stmt.query(params![query, limit as i64])?;
        let mut results = Vec::new();
        while let Some(row) = rows.next()? {
            let chunk_hash: Vec<u8> = row.get(4)?;
            if chunk_hash.len() != 32 {
                return Err(eyre!(
                    "invalid chunk_hash length for {}",
                    row.get::<_, String>(1)?
                ));
            }
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&chunk_hash);
            results.push(FtsRow {
                id: row.get(0)?,
                file_path: row.get(1)?,
                start_byte: row.get(2)?,
                end_byte: row.get(3)?,
                chunk_hash: hash,
                start_line: row.get(5)?,
                end_line: row.get(6)?,
                text: row.get(7)?,
                score: row.get(8)?,
            });
        }
        Ok(results)
    }

    fn cochange_neighbors(&self, seeds: &[String], limit: usize) -> Result<Vec<String>> {
        if seeds.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }

        let array_sql = array_value_placeholder(seeds.len());
        let sql = format!(
            "SELECT other_path FROM ( \
        SELECT dst_path AS other_path, weight \
          FROM file_cochange_edges \
         WHERE src_path IN ({array}) \
        UNION ALL \
        SELECT src_path AS other_path, weight \
          FROM file_cochange_edges \
         WHERE dst_path IN ({array}) \
      ) sq \
      GROUP BY other_path \
      ORDER BY SUM(weight) DESC \
      LIMIT ?",
            array = array_sql
        );

        let mut params_vec = Vec::with_capacity(seeds.len() * 2 + 1);
        params_vec.extend(seeds.iter().cloned().map(Value::Text));
        params_vec.extend(seeds.iter().cloned().map(Value::Text));
        params_vec.push(Value::BigInt(limit as i64));

        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(params_from_iter(params_vec))?;
        let mut results = Vec::new();
        while let Some(row) = rows.next()? {
            let path: String = row.get(0)?;
            results.push(path);
        }
        Ok(results)
    }

    fn cochange_partners(&self, file_path: &str, limit: usize) -> Result<Vec<(String, f64)>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let sql = "SELECT other_path, SUM(weight) AS total_weight FROM ( \
        SELECT dst_path AS other_path, weight \
          FROM file_cochange_edges \
         WHERE src_path = ? \
        UNION ALL \
        SELECT src_path AS other_path, weight \
          FROM file_cochange_edges \
         WHERE dst_path = ? \
      ) sq \
      GROUP BY other_path \
      ORDER BY total_weight DESC \
      LIMIT ?";

        let mut stmt = self.conn.prepare(sql)?;
        let mut rows = stmt.query(params![file_path, file_path, limit as i64])?;
        let mut results = Vec::new();
        while let Some(row) = rows.next()? {
            let path: String = row.get(0)?;
            let weight: f64 = row.get(1)?;
            results.push((path, weight));
        }
        Ok(results)
    }

    fn file_commit_count(&self, file_path: &str) -> Result<i64> {
        let count: Option<i64> = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM file_commit_edges WHERE file_path = ?",
                params![file_path],
                |row| row.get(0),
            )
            .optional()?;
        Ok(count.unwrap_or(0))
    }

    fn refresh_file_dependency_edges(&self) -> Result<()> {
        let sql = "WITH unique_symbols AS ( \
        SELECT name, MIN(file_path) AS file_path \
          FROM symbols \
         WHERE kind = 'definition' \
         GROUP BY name \
        HAVING COUNT(DISTINCT file_path) = 1 \
      ), edges AS ( \
        SELECT r.file_path AS src_path, u.file_path AS dst_path \
          FROM symbol_references r \
          JOIN unique_symbols u ON r.name = u.name \
         WHERE r.file_path <> u.file_path \
      ) \
      INSERT INTO file_dependency_edges (src_path, dst_path, reference_count) \
      SELECT src_path, dst_path, COUNT(*) AS reference_count \
        FROM edges \
       GROUP BY src_path, dst_path";

        self.with_transaction(|db| {
            db.conn.execute("DELETE FROM file_dependency_edges", [])?;
            db.conn.execute(sql, [])?;
            Ok(())
        })?;
        Ok(())
    }

    fn file_dependency_pagerank(&self, limit: usize) -> Result<Vec<(String, f64)>> {
        if limit == 0 || !self.duckpgq_loaded {
            return Ok(Vec::new());
        }

        self.refresh_file_dependency_edges()?;

        let sql = "SELECT * FROM pagerank('code_graph', 'files', 'depends_on') \
      ORDER BY 2 DESC LIMIT ?";
        let mut stmt = self.conn.prepare(sql)?;
        let mut rows = stmt.query(params![limit as i64])?;
        let mut results = Vec::new();
        while let Some(row) = rows.next()? {
            let path: String = row.get(0)?;
            let score: f64 = row.get(1)?;
            results.push((path, score));
        }
        Ok(results)
    }

    fn chunks_for_files(
        &self,
        file_paths: &[String],
        limit_per_file: usize,
    ) -> Result<Vec<SearchRow>> {
        if file_paths.is_empty() || limit_per_file == 0 {
            return Ok(Vec::new());
        }

        let mut stmt = self.conn.prepare(
            "SELECT id, file_path, start_byte, end_byte, chunk_hash, start_line, end_line, text, 0.0 AS distance \
       FROM chunks WHERE file_path = ? \
       ORDER BY ordinal ASC LIMIT ?",
        )?;

        let mut results = Vec::new();
        for file_path in file_paths {
            let mut rows = stmt.query(params![file_path, limit_per_file as i64])?;
            while let Some(row) = rows.next()? {
                let chunk_hash: Vec<u8> = row.get(4)?;
                if chunk_hash.len() != 32 {
                    return Err(eyre!(
                        "invalid chunk_hash length for {}",
                        row.get::<_, String>(1)?
                    ));
                }
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&chunk_hash);
                results.push(SearchRow {
                    id: row.get(0)?,
                    file_path: row.get(1)?,
                    start_byte: row.get(2)?,
                    end_byte: row.get(3)?,
                    chunk_hash: hash,
                    start_line: row.get(5)?,
                    end_line: row.get(6)?,
                    text: row.get(7)?,
                    distance: row.get(8)?,
                });
            }
        }

        Ok(results)
    }

    fn symbols_in_range(
        &self,
        file_path: &str,
        start_byte: i64,
        end_byte: i64,
    ) -> Result<Vec<SymbolRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, file_path, name, kind, start_byte, end_byte, language
       FROM symbols
       WHERE file_path = ?
         AND start_byte < ?
         AND end_byte > ?
       ORDER BY start_byte ASC",
        )?;
        let mut rows = stmt.query(params![file_path, end_byte, start_byte])?;
        let mut symbols = Vec::new();
        while let Some(row) = rows.next()? {
            symbols.push(SymbolRecord {
                id: row.get(0)?,
                file_path: row.get(1)?,
                name: row.get(2)?,
                kind: row.get(3)?,
                start_byte: row.get(4)?,
                end_byte: row.get(5)?,
                language: row.get(6)?,
            });
        }
        Ok(symbols)
    }
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
            chunk_hash: [0u8; 32],
            start_line: 1,
            end_line: 1,
            text: "hello".to_string(),
            kind: "text".to_string(),
            ordinal: 0,
            tokens: None,
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

        db.upsert_file("existing.rs", 12, hash, None)?;

        let record = ChunkRecord {
            id: "chunk-2".to_string(),
            file_path: "existing.rs".to_string(),
            start_byte: 10,
            end_byte: 5,
            chunk_hash: [0u8; 32],
            start_line: 1,
            end_line: 1,
            text: "broken".to_string(),
            kind: "text".to_string(),
            ordinal: 0,
            tokens: None,
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

        db.upsert_file("a.rs", 12, hash, None)?;
        db.upsert_file("b.rs", 12, hash, None)?;

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
            conn.query_row("SELECT COUNT(*) FROM reference_symbol_edges", [], |row| {
                row.get(0)
            })?;

        assert_eq!(
            edge_count, 0,
            "expected reference_symbol_edges to be removed when symbols are deleted"
        );
        Ok(())
    }
}
