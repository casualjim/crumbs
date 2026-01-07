use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rusqlite::types::Value;
use rusqlite::{Connection, OptionalExtension, params, params_from_iter};
use eyre::{Result, eyre};
use charabia::Tokenize;

use crate::repository::Repository;

pub struct Db {
    conn: Connection,
    embedding_dim: usize,
    vss_loaded: bool,
    fts_loaded: bool,
}

impl Db {
    pub fn open(path: &Path, embedding_dim: Option<usize>) -> Result<Self> {
        let conn = Connection::open(path)?;
        let mut db = Self {
            conn,
            embedding_dim: 0,
            vss_loaded: false,
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
                |row| row.get::<_, String>(0),
            )
            .optional()?;

        let resolved_dim = match (existing_dim, embedding_dim) {
            (Some(value), Some(requested)) => {
                let parsed: usize = value
                    .parse::<usize>()
                    .map_err(|err: std::num::ParseIntError| eyre!(err))?;
                if parsed != requested {
                    return Err(eyre!(
                        "embedding dimension mismatch: db has {}, requested {}",
                        parsed,
                        requested
                    ));
                }
                parsed
            }
            (Some(value), None) => value
                .parse::<usize>()
                .map_err(|err: std::num::ParseIntError| eyre!(err))?,
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
        self.vss_loaded = true;
        self.fts_loaded = false;

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
          updated_at TEXT DEFAULT CURRENT_TIMESTAMP
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
          fts_text TEXT NOT NULL,
          kind TEXT NOT NULL,
          ordinal INTEGER NOT NULL,
          tokens BLOB,
          embedding F32_BLOB({dim}) NOT NULL
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
        CREATE INDEX IF NOT EXISTS symbols_name_kind_idx ON symbols (name, kind);
        CREATE INDEX IF NOT EXISTS symbols_name_file_idx ON symbols (name, file_path);
        CREATE INDEX IF NOT EXISTS references_file_path_idx ON symbol_references (file_path);
        CREATE INDEX IF NOT EXISTS references_name_idx ON symbol_references (name);
        CREATE INDEX IF NOT EXISTS references_name_file_idx ON symbol_references (name, file_path);",
            dim = self.embedding_dim
        );

        self.conn.execute_batch(&create_sql)?;
        self.ensure_fts_text_column()?;
        self.conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_idx \
             ON chunks(libsql_vector_idx(embedding));",
        )?;
        self.init_fts()?;
        Ok(())
    }

    fn init_fts(&mut self) -> Result<()> {
        let fts_sql = "CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5( \
          fts_text, content='chunks', content_rowid='rowid' \
        ); \
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN \
          INSERT INTO fts_chunks(rowid, fts_text) VALUES (new.rowid, new.fts_text); \
        END; \
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN \
          INSERT INTO fts_chunks(fts_chunks, rowid, fts_text) VALUES('delete', old.rowid, old.fts_text); \
        END; \
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN \
          INSERT INTO fts_chunks(fts_chunks, rowid, fts_text) VALUES('delete', old.rowid, old.fts_text); \
          INSERT INTO fts_chunks(rowid, fts_text) VALUES (new.rowid, new.fts_text); \
        END;";
        if let Err(err) = self.conn.execute_batch(fts_sql) {
            tracing::warn!("failed to initialize FTS: {err}");
            if let Err(rebuild_err) = self.rebuild_fts_schema(fts_sql) {
                self.fts_loaded = false;
                tracing::warn!("failed to rebuild FTS schema: {rebuild_err}");
                return Ok(());
            }
        }
        if let Err(err) = self
            .conn
            .execute("INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild')", [])
        {
            tracing::warn!("failed to rebuild FTS index: {err}");
        }
        self.fts_loaded = true;
        Ok(())
    }

    fn rebuild_fts_schema(&self, fts_sql: &str) -> Result<()> {
        self.conn.execute_batch(
            "DROP TRIGGER IF EXISTS chunks_ai; \
             DROP TRIGGER IF EXISTS chunks_ad; \
             DROP TRIGGER IF EXISTS chunks_au; \
             DROP TABLE IF EXISTS fts_chunks;",
        )?;
        self.conn.execute_batch(fts_sql)?;
        Ok(())
    }

    fn ensure_fts_text_column(&self) -> Result<()> {
        let mut stmt = self.conn.prepare("PRAGMA table_info(chunks)")?;
        let mut rows = stmt.query([])?;
        let mut has_column = false;
        while let Some(row) = rows.next()? {
            let name: String = row.get(1)?;
            if name == "fts_text" {
                has_column = true;
                break;
            }
        }

        if !has_column {
            self.conn.execute(
                "ALTER TABLE chunks ADD COLUMN fts_text TEXT NOT NULL DEFAULT ''",
                [],
            )?;
        }

        self.conn
            .execute("UPDATE chunks SET fts_text = text WHERE fts_text = ''", [])?;
        Ok(())
    }

    fn clear_file_chunks(&self, file_path: &str) -> Result<()> {
        self.delete_file_chunk_rows(file_path)
    }

    fn clear_file_graph(&self, file_path: &str) -> Result<()> {
        self.delete_file_graph_rows(file_path)
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

    fn delete_file_chunk_rows(&self, file_path: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM file_chunk_edges WHERE file_path = ?",
            params![file_path],
        )?;
        self.conn
            .execute("DELETE FROM chunks WHERE file_path = ?", params![file_path])?;
        Ok(())
    }

    fn delete_file_graph_rows(&self, file_path: &str) -> Result<()> {
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
            "DELETE FROM file_reference_edges WHERE reference_id IN \
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
            "DELETE FROM file_symbol_edges WHERE symbol_id IN \
       (SELECT id FROM symbols WHERE file_path = ?)",
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

    fn upsert_file(
        &self,
        file_path: &str,
        file_size: u64,
        content_hash: [u8; 32],
        primary_language: Option<&str>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO files (path, content_hash, size, primary_language, updated_at)
       VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
       ON CONFLICT(path) DO UPDATE SET
         content_hash = excluded.content_hash,
         size = excluded.size,
         primary_language = excluded.primary_language,
         updated_at = CURRENT_TIMESTAMP
       WHERE files.content_hash IS NULL
          OR files.content_hash != excluded.content_hash
          OR files.size != excluded.size
          OR COALESCE(files.primary_language, '') != COALESCE(excluded.primary_language, '')",
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
       VALUES (?, ?, ?, CURRENT_TIMESTAMP)
       ON CONFLICT(path) DO UPDATE SET
         size = excluded.size,
         primary_language = excluded.primary_language,
         updated_at = CURRENT_TIMESTAMP
       WHERE files.size != excluded.size
          OR COALESCE(files.primary_language, '') != COALESCE(excluded.primary_language, '')",
            params![file_path, file_size as i64, primary_language],
        )?;
        Ok(())
    }

    #[cfg(test)]
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

        let tokens_blob = record
            .tokens
            .as_ref()
            .filter(|tokens| !tokens.is_empty())
            .map(|tokens| encode_u32_blob(tokens));
        let tokens_sql = if tokens_blob.is_some() { "?" } else { "NULL" };
        let embedding_json = encode_f32_json(embedding);
        let sql = format!(
            "INSERT INTO chunks \
      (id, file_path, start_byte, end_byte, chunk_hash, start_line, end_line, text, fts_text, kind, ordinal, tokens, embedding) \
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {tokens_sql}, vector32(?))",
            tokens_sql = tokens_sql,
        );

        let mut params_vec = Vec::with_capacity(13);
        params_vec.push(Value::Text(record.id.clone()));
        params_vec.push(Value::Text(record.file_path.clone()));
        params_vec.push(Value::Integer(record.start_byte as i64));
        params_vec.push(Value::Integer(record.end_byte as i64));
        params_vec.push(Value::Blob(record.chunk_hash.to_vec()));
        params_vec.push(Value::Integer(record.start_line as i64));
        params_vec.push(Value::Integer(record.end_line as i64));
        params_vec.push(Value::Text(record.text.clone()));
        params_vec.push(Value::Text(record.fts_text.clone()));
        params_vec.push(Value::Text(record.kind.clone()));
        params_vec.push(Value::Integer(record.ordinal as i64));
        if let Some(tokens_blob) = tokens_blob {
            params_vec.push(Value::Blob(tokens_blob));
        }
        params_vec.push(Value::Text(embedding_json));

        let mut stmt = self.conn.prepare(&sql)?;
        stmt.execute(params_from_iter(params_vec))?;

        self.conn.execute(
            "INSERT INTO file_chunk_edges (file_path, chunk_id, ordinal) VALUES (?, ?, ?)",
            params![record.file_path, record.id, record.ordinal as i32],
        )?;

        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn upsert_chunk_with_embedding(
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

        let tokens_blob = record
            .tokens
            .as_ref()
            .filter(|tokens| !tokens.is_empty())
            .map(|tokens| encode_u32_blob(tokens));
        let tokens_sql = if tokens_blob.is_some() { "?" } else { "NULL" };
        let embedding_json = encode_f32_json(embedding);
        let sql = format!(
            "INSERT INTO chunks \
      (id, file_path, start_byte, end_byte, chunk_hash, start_line, end_line, text, fts_text, kind, ordinal, tokens, embedding) \
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {tokens_sql}, vector32(?)) \
      ON CONFLICT(id) DO UPDATE SET \
        file_path = excluded.file_path, \
        start_byte = excluded.start_byte, \
        end_byte = excluded.end_byte, \
        chunk_hash = excluded.chunk_hash, \
        start_line = excluded.start_line, \
        end_line = excluded.end_line, \
        text = excluded.text, \
        fts_text = excluded.fts_text, \
        kind = excluded.kind, \
        ordinal = excluded.ordinal, \
        tokens = excluded.tokens, \
        embedding = excluded.embedding",
            tokens_sql = tokens_sql,
        );

        let mut params_vec = Vec::with_capacity(13);
        params_vec.push(Value::Text(record.id.clone()));
        params_vec.push(Value::Text(record.file_path.clone()));
        params_vec.push(Value::Integer(record.start_byte as i64));
        params_vec.push(Value::Integer(record.end_byte as i64));
        params_vec.push(Value::Blob(record.chunk_hash.to_vec()));
        params_vec.push(Value::Integer(record.start_line as i64));
        params_vec.push(Value::Integer(record.end_line as i64));
        params_vec.push(Value::Text(record.text.clone()));
        params_vec.push(Value::Text(record.fts_text.clone()));
        params_vec.push(Value::Text(record.kind.clone()));
        params_vec.push(Value::Integer(record.ordinal as i64));
        if let Some(tokens_blob) = tokens_blob {
            params_vec.push(Value::Blob(tokens_blob));
        }
        params_vec.push(Value::Text(embedding_json));

        self.with_transaction(|db| {
            let mut stmt = db.conn.prepare(&sql)?;
            stmt.execute(params_from_iter(params_vec))?;

            db.conn.execute(
                "INSERT INTO file_chunk_edges (file_path, chunk_id, ordinal) VALUES (?, ?, ?) \
                 ON CONFLICT(file_path, chunk_id) DO UPDATE SET ordinal = excluded.ordinal",
                params![record.file_path, record.id, record.ordinal as i32],
            )?;
            Ok(())
        })?;

        Ok(())
    }

    fn upsert_chunks_with_embeddings(
        &self,
        records: &[ChunkRecord],
        embeddings: &[Vec<f32>],
    ) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        if records.len() != embeddings.len() {
            return Err(eyre!(
                "embedding count mismatch: {} records, {} embeddings",
                records.len(),
                embeddings.len()
            ));
        }

        let mut values_sql = Vec::with_capacity(records.len());
        let mut values_edges = Vec::with_capacity(records.len());
        let mut params_vec = Vec::new();
        let mut params_edges = Vec::with_capacity(records.len() * 3);
        for (record, embedding) in records.iter().zip(embeddings.iter()) {
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

            let tokens_blob = record
                .tokens
                .as_ref()
                .filter(|tokens| !tokens.is_empty())
                .map(|tokens| encode_u32_blob(tokens));
            let tokens_sql = if tokens_blob.is_some() { "?" } else { "NULL" };
            let embed_sql = "vector32(?)";

            values_sql.push(format!(
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {tokens_sql}, {embed_sql})",
                tokens_sql = tokens_sql,
                embed_sql = embed_sql
            ));
            values_edges.push("(?, ?, ?)".to_string());

            params_vec.push(Value::Text(record.id.clone()));
            params_vec.push(Value::Text(record.file_path.clone()));
            params_vec.push(Value::Integer(record.start_byte as i64));
            params_vec.push(Value::Integer(record.end_byte as i64));
            params_vec.push(Value::Blob(record.chunk_hash.to_vec()));
            params_vec.push(Value::Integer(record.start_line as i64));
            params_vec.push(Value::Integer(record.end_line as i64));
            params_vec.push(Value::Text(record.text.clone()));
            params_vec.push(Value::Text(record.fts_text.clone()));
            params_vec.push(Value::Text(record.kind.clone()));
            params_vec.push(Value::Integer(record.ordinal as i64));
            if let Some(tokens_blob) = tokens_blob {
                params_vec.push(Value::Blob(tokens_blob));
            }
            params_vec.push(Value::Text(encode_f32_json(embedding)));

            params_edges.push(Value::Text(record.file_path.clone()));
            params_edges.push(Value::Text(record.id.clone()));
            params_edges.push(Value::Integer(record.ordinal as i64));
        }

        let sql = format!(
            "INSERT INTO chunks (id, file_path, start_byte, end_byte, chunk_hash, start_line, end_line, text, fts_text, kind, ordinal, tokens, embedding) \
             VALUES {values} \
             ON CONFLICT(id) DO UPDATE SET \
               file_path = excluded.file_path, \
               start_byte = excluded.start_byte, \
               end_byte = excluded.end_byte, \
               chunk_hash = excluded.chunk_hash, \
               start_line = excluded.start_line, \
               end_line = excluded.end_line, \
               text = excluded.text, \
               fts_text = excluded.fts_text, \
               kind = excluded.kind, \
               ordinal = excluded.ordinal, \
               tokens = excluded.tokens, \
               embedding = excluded.embedding",
            values = values_sql.join(", ")
        );

        let mut stmt = self.conn.prepare(&sql)?;
        stmt.execute(params_from_iter(params_vec))?;
        let edges_sql = format!(
            "INSERT OR REPLACE INTO file_chunk_edges (file_path, chunk_id, ordinal) \
             VALUES {values}",
            values = values_edges.join(", ")
        );
        let mut edges_stmt = self.conn.prepare(&edges_sql)?;
        edges_stmt.execute(params_from_iter(params_edges))?;
        Ok(())
    }

    fn update_chunk_without_embedding(&self, record: &ChunkRecord) -> Result<()> {
        self.ensure_file_exists(&record.file_path)?;
        let tokens_blob = record
            .tokens
            .as_ref()
            .filter(|tokens| !tokens.is_empty())
            .map(|tokens| encode_u32_blob(tokens));
        let tokens_sql = if tokens_blob.is_some() { "?" } else { "NULL" };
        let sql = format!(
            "UPDATE chunks SET \
           file_path = ?, start_byte = ?, end_byte = ?, chunk_hash = ?, start_line = ?, end_line = ?, \
           text = ?, fts_text = ?, kind = ?, ordinal = ?, tokens = {tokens_sql} \
           WHERE id = ?",
            tokens_sql = tokens_sql
        );
        let mut params_vec = Vec::with_capacity(12);
        params_vec.push(Value::Text(record.file_path.clone()));
        params_vec.push(Value::Integer(record.start_byte as i64));
        params_vec.push(Value::Integer(record.end_byte as i64));
        params_vec.push(Value::Blob(record.chunk_hash.to_vec()));
        params_vec.push(Value::Integer(record.start_line as i64));
        params_vec.push(Value::Integer(record.end_line as i64));
        params_vec.push(Value::Text(record.text.clone()));
        params_vec.push(Value::Text(record.fts_text.clone()));
        params_vec.push(Value::Text(record.kind.clone()));
        params_vec.push(Value::Integer(record.ordinal as i64));
        if let Some(tokens_blob) = tokens_blob {
            params_vec.push(Value::Blob(tokens_blob));
        }
        params_vec.push(Value::Text(record.id.clone()));
        self.conn.execute(
            "DELETE FROM file_chunk_edges WHERE chunk_id = ?",
            params![record.id],
        )?;

        let mut stmt = self.conn.prepare(&sql)?;
        let updated = stmt.execute(params_from_iter(params_vec))?;
        if updated == 0 {
            return Err(eyre!("missing existing chunk {}", record.id));
        }

        self.conn.execute(
            "INSERT INTO file_chunk_edges (file_path, chunk_id, ordinal) VALUES (?, ?, ?) \
             ON CONFLICT(file_path, chunk_id) DO UPDATE SET ordinal = excluded.ordinal",
            params![record.file_path, record.id, record.ordinal as i32],
        )?;
        Ok(())
    }

    #[cfg(test)]
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

    #[cfg(test)]
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

    #[cfg(test)]
    fn link_reference_symbol(&self, reference_id: &str, symbol_id: &str) -> Result<()> {
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
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        if exists.is_none() {
            return Err(eyre!("file not found: {}", file_path));
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
                    return Err(eyre!(err));
                }
                Ok(value)
            }
            Err(err) => {
                let _ = self.conn.execute_batch("ROLLBACK");
                Err(err)
            }
        }
    }

}

#[derive(Clone)]
pub struct ChunkRecord {
    pub id: String,
    pub file_path: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub chunk_hash: [u8; 32],
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub fts_text: String,
    pub kind: String,
    pub ordinal: usize,
    pub tokens: Option<Vec<u32>>,
}

#[derive(Clone)]
pub struct SymbolRecord {
    pub id: String,
    pub file_path: String,
    pub name: String,
    pub kind: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub language: String,
}

#[derive(Clone)]
pub struct ReferenceRecord {
    pub id: String,
    pub file_path: String,
    pub name: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub language: String,
}

#[derive(Clone)]
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
    vars
}

pub(crate) fn build_fts_text(text: &str) -> String {
    let terms = tokenize_for_fts(text);
    if terms.is_empty() {
        String::new()
    } else {
        terms.join(" ")
    }
}

pub(crate) fn build_fts_query(text: &str) -> Option<String> {
    let mut terms = Vec::new();
    let mut seen = std::collections::HashSet::<String>::new();
    for term in tokenize_for_fts(text) {
        if term.len() < 2 {
            continue;
        }
        if seen.insert(term.clone()) {
            terms.push(term);
        }
        if terms.len() >= 32 {
            break;
        }
    }

    if terms.is_empty() {
        None
    } else {
        Some(terms.join(" OR "))
    }
}

fn tokenize_for_fts(text: &str) -> Vec<String> {
    let mut terms = Vec::new();
    for token in text.tokenize() {
        if !token.is_word() {
            continue;
        }
        let lemma = token.lemma();
        if lemma.is_empty() {
            continue;
        }
        terms.push(lemma.to_string());
        for part in split_identifier(lemma) {
            if part != lemma {
                terms.push(part);
            }
        }
    }
    terms
}

fn split_identifier(token: &str) -> Vec<String> {
    #[derive(Copy, Clone, PartialEq, Eq)]
    enum Kind {
        Lower,
        Upper,
        Digit,
        Other,
    }

    let mut parts = Vec::new();
    let mut current = String::new();
    let mut prev_kind = None;

    for ch in token.chars() {
        let kind = if ch.is_ascii_lowercase() {
            Kind::Lower
        } else if ch.is_ascii_uppercase() {
            Kind::Upper
        } else if ch.is_ascii_digit() {
            Kind::Digit
        } else {
            Kind::Other
        };

        if let Some(prev) = prev_kind {
            let boundary = matches!(
                (prev, kind),
                (Kind::Lower, Kind::Upper)
                    | (Kind::Lower, Kind::Digit)
                    | (Kind::Upper, Kind::Digit)
                    | (Kind::Digit, Kind::Lower)
                    | (Kind::Digit, Kind::Upper)
            );

            if boundary && !current.is_empty() {
                if prev == Kind::Upper && kind == Kind::Lower && current.len() > 1 {
                    let last = current.pop().unwrap();
                    if !current.is_empty() {
                        parts.push(current.clone());
                    }
                    current.clear();
                    current.push(last.to_ascii_lowercase());
                } else {
                    parts.push(current.clone());
                    current.clear();
                }
            }
        }

        if kind == Kind::Other {
            if !current.is_empty() {
                parts.push(current.clone());
                current.clear();
            }
        } else {
            current.push(ch.to_ascii_lowercase());
        }
        prev_kind = Some(kind);
    }

    if !current.is_empty() {
        parts.push(current);
    }

    parts
}

fn encode_u32_blob(tokens: &[u32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(tokens.len() * 4);
    for token in tokens {
        buf.extend_from_slice(&token.to_le_bytes());
    }
    buf
}

fn encode_f32_json(embedding: &[f32]) -> String {
    let mut out = String::with_capacity(embedding.len() * 8 + 2);
    out.push('[');
    for (idx, value) in embedding.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
    out
}

impl Repository for Db {
    fn load_existing_hashes(&self) -> Result<BTreeMap<PathBuf, [u8; 32]>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path, content_hash FROM files WHERE content_hash IS NOT NULL")?;
        let mut rows = stmt.query([])?;
        let mut hashes = BTreeMap::new();
        while let Some(row) = rows.next()? {
            let path: String = row.get::<_, String>(0)?;
            let hash: Vec<u8> = row.get::<_, Vec<u8>>(1)?;
            if hash.len() != 32 {
                continue;
            }
            let mut buf = [0u8; 32];
            buf.copy_from_slice(&hash);
            hashes.insert(PathBuf::from(path), buf);
        }
        Ok(hashes)
    }

    fn find_chunk_id(
        &self,
        _file_path: &str,
        _start_byte: usize,
        _end_byte: usize,
        kind: &str,
        chunk_hash: [u8; 32],
    ) -> Result<Option<String>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id FROM chunks \
                 WHERE kind = ? AND chunk_hash = ? \
                 LIMIT 1",
            )?;
        let id: Option<String> = stmt
            .query_row(
                params![kind, chunk_hash.to_vec()],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        Ok(id)
    }

    fn upsert_file_metadata(
        &self,
        file_path: &str,
        file_size: u64,
        content_hash: [u8; 32],
        primary_language: Option<String>,
    ) -> Result<()> {
        self.upsert_file(
            file_path,
            file_size,
            content_hash,
            primary_language.as_deref(),
        )
    }

    fn ensure_file_row(
        &self,
        file_path: &str,
        file_size: u64,
        primary_language: Option<String>,
    ) -> Result<()> {
        self.ensure_file(file_path, file_size, primary_language.as_deref())
    }

    fn upsert_chunks_with_embeddings(
        &self,
        records: &[ChunkRecord],
        embeddings: &[Vec<f32>],
    ) -> Result<()> {
        self.upsert_chunks_with_embeddings(records, embeddings)
    }

    fn update_chunk_without_embedding(&self, record: &ChunkRecord) -> Result<()> {
        self.update_chunk_without_embedding(record)
    }

    fn delete_missing_chunks(&self, file_path: &str, keep_ids: &[String]) -> Result<()> {
        if keep_ids.is_empty() {
            return self.clear_file_chunks(file_path);
        }

        let placeholders = "?,".repeat(keep_ids.len()).trim_end_matches(',').to_string();
        let delete_edges = format!(
            "DELETE FROM file_chunk_edges WHERE file_path = ? AND chunk_id NOT IN ({})",
            placeholders
        );
        let delete_chunks = format!(
            "DELETE FROM chunks WHERE file_path = ? AND id NOT IN ({})",
            placeholders
        );

        self.with_transaction(|db| {
            let mut params = Vec::with_capacity(keep_ids.len() + 1);
            params.push(Value::Text(file_path.to_string()));
            params.extend(keep_ids.iter().cloned().map(Value::Text));
            db.conn
                .execute(&delete_edges, params_from_iter(params.clone()))?;
            db.conn.execute(&delete_chunks, params_from_iter(params))?;
            Ok(())
        })?;
        Ok(())
    }

    fn upsert_file_graph(
        &self,
        file_path: &str,
        file_size: u64,
        content_hash: [u8; 32],
        language: &str,
        primary_language: Option<String>,
        graph: GraphData,
    ) -> Result<()> {
        let primary_language = primary_language.as_deref();
        if graph.symbols.is_empty() && graph.references.is_empty() && graph.resolutions.is_empty() {
            self.ensure_file(file_path, file_size, primary_language)?;
            self.delete_file_graph_rows(file_path)?;
            self.upsert_file(file_path, file_size, content_hash, primary_language)?;
            self.update_file_dependency_edges(file_path)?;
            return Ok(());
        }

        self.with_transaction(|db| {
            db.ensure_file(file_path, file_size, primary_language)?;

            let symbol_ids: Vec<String> = graph.symbols.iter().map(|s| s.id.clone()).collect();
            let reference_ids: Vec<String> =
                graph.references.iter().map(|r| r.id.clone()).collect();

            db.conn.execute(
                "DELETE FROM reference_symbol_edges WHERE symbol_id IN \
                 (SELECT id FROM symbols WHERE file_path = ?)",
                params![file_path],
            )?;
            db.conn.execute(
                "DELETE FROM reference_symbol_edges WHERE reference_id IN \
                 (SELECT id FROM symbol_references WHERE file_path = ?)",
                params![file_path],
            )?;
            db.conn.execute(
                "DELETE FROM file_symbol_edges WHERE symbol_id IN \
                 (SELECT id FROM symbols WHERE file_path = ?)",
                params![file_path],
            )?;
            db.conn.execute(
                "DELETE FROM file_reference_edges WHERE reference_id IN \
                 (SELECT id FROM symbol_references WHERE file_path = ?)",
                params![file_path],
            )?;

            if symbol_ids.is_empty() {
                db.conn.execute(
                    "DELETE FROM file_symbol_edges WHERE file_path = ?",
                    params![file_path],
                )?;
                db.conn
                    .execute("DELETE FROM symbols WHERE file_path = ?", params![file_path])?;
            } else {
                let placeholders =
                    "?,".repeat(symbol_ids.len()).trim_end_matches(',').to_string();
                let mut params = Vec::with_capacity(symbol_ids.len() + 1);
                params.push(Value::Text(file_path.to_string()));
                params.extend(symbol_ids.iter().cloned().map(Value::Text));
                db.conn.execute(
                    &format!(
                        "DELETE FROM file_symbol_edges WHERE file_path = ? AND symbol_id NOT IN ({})",
                        placeholders
                    ),
                    params_from_iter(params.clone()),
                )?;
                db.conn.execute(
                    &format!(
                        "DELETE FROM symbols WHERE file_path = ? AND id NOT IN ({})",
                        placeholders
                    ),
                    params_from_iter(params),
                )?;
            }

            if reference_ids.is_empty() {
                db.conn.execute(
                    "DELETE FROM file_reference_edges WHERE file_path = ?",
                    params![file_path],
                )?;
                db.conn.execute(
                    "DELETE FROM symbol_references WHERE file_path = ?",
                    params![file_path],
                )?;
            } else {
                let placeholders =
                    "?,".repeat(reference_ids.len()).trim_end_matches(',').to_string();
                let mut params = Vec::with_capacity(reference_ids.len() + 1);
                params.push(Value::Text(file_path.to_string()));
                params.extend(reference_ids.iter().cloned().map(Value::Text));
                db.conn.execute(
                    &format!(
                        "DELETE FROM file_reference_edges WHERE file_path = ? AND reference_id NOT IN ({})",
                        placeholders
                    ),
                    params_from_iter(params.clone()),
                )?;
                db.conn.execute(
                    &format!(
                        "DELETE FROM symbol_references WHERE file_path = ? AND id NOT IN ({})",
                        placeholders
                    ),
                    params_from_iter(params),
                )?;
            }

            if !graph.symbols.is_empty() {
                let mut values = Vec::with_capacity(graph.symbols.len());
                let mut params = Vec::with_capacity(graph.symbols.len() * 7);
                let mut edge_values = Vec::with_capacity(graph.symbols.len());
                let mut edge_params = Vec::with_capacity(graph.symbols.len() * 2);
                for symbol in &graph.symbols {
                    let symbol_file_path = if symbol.file_path.is_empty() {
                        file_path
                    } else {
                        symbol.file_path.as_str()
                    };
                    let symbol_language = if symbol.language.is_empty() {
                        language
                    } else {
                        symbol.language.as_str()
                    };
                    values.push("(?, ?, ?, ?, ?, ?, ?)".to_string());
                    edge_values.push("(?, ?)".to_string());
                    params.push(Value::Text(symbol.id.clone()));
                    params.push(Value::Text(symbol_file_path.to_string()));
                    params.push(Value::Text(symbol.name.clone()));
                    params.push(Value::Text(symbol.kind.clone()));
                    params.push(Value::Integer(symbol.start_byte as i64));
                    params.push(Value::Integer(symbol.end_byte as i64));
                    params.push(Value::Text(symbol_language.to_string()));
                    edge_params.push(Value::Text(symbol_file_path.to_string()));
                    edge_params.push(Value::Text(symbol.id.clone()));
                }

                let sql = format!(
                    "INSERT INTO symbols (id, file_path, name, kind, start_byte, end_byte, language) \
                     VALUES {values} \
                     ON CONFLICT(id) DO UPDATE SET \
                       file_path = excluded.file_path, \
                       name = excluded.name, \
                       kind = excluded.kind, \
                       start_byte = excluded.start_byte, \
                       end_byte = excluded.end_byte, \
                       language = excluded.language",
                    values = values.join(", ")
                );
                let mut stmt = db.conn.prepare(&sql)?;
                stmt.execute(params_from_iter(params))?;

                let edge_sql = format!(
                    "INSERT INTO file_symbol_edges (file_path, symbol_id) \
                     VALUES {values} \
                     ON CONFLICT(file_path, symbol_id) DO NOTHING",
                    values = edge_values.join(", ")
                );
                let mut edge_stmt = db.conn.prepare(&edge_sql)?;
                edge_stmt.execute(params_from_iter(edge_params))?;
            }

            if !graph.references.is_empty() {
                let mut values = Vec::with_capacity(graph.references.len());
                let mut params = Vec::with_capacity(graph.references.len() * 6);
                let mut edge_values = Vec::with_capacity(graph.references.len());
                let mut edge_params = Vec::with_capacity(graph.references.len() * 2);
                for reference in &graph.references {
                    let reference_file_path = if reference.file_path.is_empty() {
                        file_path
                    } else {
                        reference.file_path.as_str()
                    };
                    let reference_language = if reference.language.is_empty() {
                        language
                    } else {
                        reference.language.as_str()
                    };
                    values.push("(?, ?, ?, ?, ?, ?)".to_string());
                    edge_values.push("(?, ?)".to_string());
                    params.push(Value::Text(reference.id.clone()));
                    params.push(Value::Text(reference_file_path.to_string()));
                    params.push(Value::Text(reference.name.clone()));
                    params.push(Value::Integer(reference.start_byte as i64));
                    params.push(Value::Integer(reference.end_byte as i64));
                    params.push(Value::Text(reference_language.to_string()));
                    edge_params.push(Value::Text(reference_file_path.to_string()));
                    edge_params.push(Value::Text(reference.id.clone()));
                }

                let sql = format!(
                    "INSERT INTO symbol_references (id, file_path, name, start_byte, end_byte, language) \
                     VALUES {values} \
                     ON CONFLICT(id) DO UPDATE SET \
                       file_path = excluded.file_path, \
                       name = excluded.name, \
                       start_byte = excluded.start_byte, \
                       end_byte = excluded.end_byte, \
                       language = excluded.language",
                    values = values.join(", ")
                );
                let mut stmt = db.conn.prepare(&sql)?;
                stmt.execute(params_from_iter(params))?;

                let edge_sql = format!(
                    "INSERT INTO file_reference_edges (file_path, reference_id) \
                     VALUES {values} \
                     ON CONFLICT(file_path, reference_id) DO NOTHING",
                    values = edge_values.join(", ")
                );
                let mut edge_stmt = db.conn.prepare(&edge_sql)?;
                edge_stmt.execute(params_from_iter(edge_params))?;
            }

            if !graph.symbols.is_empty() && !graph.references.is_empty() {
                let mut symbol_values = Vec::with_capacity(graph.symbols.len());
                let mut symbol_params = Vec::with_capacity(graph.symbols.len() * 2);
                for symbol in &graph.symbols {
                    symbol_values.push("(?, ?)".to_string());
                    symbol_params.push(Value::Text(symbol.id.clone()));
                    symbol_params.push(Value::Text(symbol.name.clone()));
                }

                let mut ref_values = Vec::with_capacity(graph.references.len());
                let mut ref_params = Vec::with_capacity(graph.references.len() * 2);
                for reference in &graph.references {
                    ref_values.push("(?, ?)".to_string());
                    ref_params.push(Value::Text(reference.id.clone()));
                    ref_params.push(Value::Text(reference.name.clone()));
                }

                let sql = format!(
                    "WITH symbols(id, name) AS (SELECT * FROM (VALUES {sym_values})), \
                     refs(id, name) AS (SELECT * FROM (VALUES {ref_values})), \
                     unique_links AS ( \
                       SELECT r.id AS reference_id, MIN(s.id) AS symbol_id \
                       FROM refs r \
                       JOIN symbols s ON r.name = s.name \
                       GROUP BY r.id \
                       HAVING COUNT(DISTINCT s.id) = 1 \
                     ) \
                     INSERT OR IGNORE INTO reference_symbol_edges (reference_id, symbol_id) \
                     SELECT reference_id, symbol_id FROM unique_links",
                    sym_values = symbol_values.join(", "),
                    ref_values = ref_values.join(", ")
                );
                let mut params = Vec::with_capacity(symbol_params.len() + ref_params.len());
                params.extend(symbol_params);
                params.extend(ref_params);
                let mut stmt = db.conn.prepare(&sql)?;
                stmt.execute(params_from_iter(params))?;
            }

            if !graph.resolutions.is_empty() {
                let mut values = Vec::with_capacity(graph.resolutions.len());
                let mut params = Vec::with_capacity(graph.resolutions.len() * 2);
                for (reference_id, symbol_id) in &graph.resolutions {
                    values.push("(?, ?)".to_string());
                    params.push(Value::Text(reference_id.clone()));
                    params.push(Value::Text(symbol_id.clone()));
                }
                let sql = format!(
                    "INSERT OR IGNORE INTO reference_symbol_edges (reference_id, symbol_id) \
                     VALUES {values}",
                    values = values.join(", ")
                );
                let mut stmt = db.conn.prepare(&sql)?;
                stmt.execute(params_from_iter(params))?;
            }

            db.upsert_file(file_path, file_size, content_hash, primary_language)?;
            Ok(())
        })?;

        self.update_file_dependency_edges(file_path)?;
        Ok(())
    }

    fn delete_file(&self, file_path: &str) -> Result<()> {
        self.clear_file_graph(file_path)?;
        self.clear_file_chunks(file_path)?;
        self.clear_file_history(file_path)?;
        self.conn
            .execute("DELETE FROM files WHERE path = ?", params![file_path])?;
        Ok(())
    }

    fn list_files(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT path FROM files")?;
        let mut rows = stmt.query([])?;
        let mut files = Vec::new();
        while let Some(row) = rows.next()? {
            let path: String = row.get::<_, String>(0)?;
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
            let lang: Option<String> = row.get::<_, Option<String>>(0)?;
            Ok(lang)
        } else {
            Ok(None)
        }
    }

    fn upsert_history_edges(
        &self,
        file_commit_edges: &[(String, String)],
        cochange_edges: &[(String, String, i64, f64)],
    ) -> Result<()> {
        self.with_transaction(|db| {
            for (file_path, commit_id) in file_commit_edges {
                db.conn.execute(
                    "INSERT INTO file_commit_edges (file_path, commit_id) VALUES (?, ?) \
                     ON CONFLICT(file_path, commit_id) DO NOTHING",
                    params![file_path, commit_id],
                )?;
            }

            for (src, dst, commit_count, weight) in cochange_edges {
                db.conn.execute(
                    "INSERT INTO file_cochange_edges \
                     (src_path, dst_path, commit_count, weight) VALUES (?, ?, ?, ?) \
                     ON CONFLICT(src_path, dst_path) DO UPDATE SET \
                       commit_count = excluded.commit_count, \
                       weight = excluded.weight",
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
        if !self.vss_loaded || limit == 0 {
            return Ok(Vec::new());
        }
        if query_embedding.len() != self.embedding_dim {
            return Err(eyre!(
                "embedding dimension mismatch: expected {}, got {}",
                self.embedding_dim,
                query_embedding.len()
            ));
        }
        let query_json = encode_f32_json(query_embedding);
        let sql = "SELECT c.id, c.file_path, c.start_byte, c.end_byte, c.chunk_hash, c.start_line, \
       c.end_line, c.text, vector_distance_cos(c.embedding, vector32(?)) AS distance \
       FROM vector_top_k('chunks_embedding_idx', vector32(?), ?) v \
       JOIN chunks c ON c.rowid = v.id \
       ORDER BY distance ASC";

        let mut stmt = self.conn.prepare(sql)?;
        let mut rows = stmt.query(params![query_json, query_json, limit as i64])?;
        let mut results = Vec::new();
        while let Some(row) = rows.next()? {
            let chunk_hash: Vec<u8> = row.get::<_, Vec<u8>>(4)?;
            if chunk_hash.len() != 32 {
                return Err(eyre!(
                    "invalid chunk_hash length for {}",
                    row.get::<_, String>(1)?
                ));
            }
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&chunk_hash);
            results.push(SearchRow {
                id: row.get::<_, String>(0)?,
                file_path: row.get::<_, String>(1)?,
                start_byte: row.get::<_, i64>(2)?,
                end_byte: row.get::<_, i64>(3)?,
                chunk_hash: hash,
                start_line: row.get::<_, i64>(5)?,
                end_line: row.get::<_, i64>(6)?,
                text: row.get::<_, String>(7)?,
                distance: row.get::<_, f64>(8)?,
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

        let Some(query) = build_fts_query(query) else {
            return Ok(Vec::new());
        };

        let sql = "SELECT c.id, c.file_path, c.start_byte, c.end_byte, c.chunk_hash, c.start_line, \
       c.end_line, c.text, (1.0 / (1.0 + bm25(fts_chunks))) AS score \
      FROM fts_chunks \
      JOIN chunks c ON c.rowid = fts_chunks.rowid \
      WHERE fts_chunks MATCH ? \
      ORDER BY score DESC \
      LIMIT ?";

        let mut stmt = self.conn.prepare(sql)?;
        let mut rows = stmt.query(params![query, limit as i64])?;
        let mut results = Vec::new();
        while let Some(row) = rows.next()? {
            let chunk_hash: Vec<u8> = row.get::<_, Vec<u8>>(4)?;
            if chunk_hash.len() != 32 {
                return Err(eyre!(
                    "invalid chunk_hash length for {}",
                    row.get::<_, String>(1)?
                ));
            }
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&chunk_hash);
            results.push(FtsRow {
                id: row.get::<_, String>(0)?,
                file_path: row.get::<_, String>(1)?,
                start_byte: row.get::<_, i64>(2)?,
                end_byte: row.get::<_, i64>(3)?,
                chunk_hash: hash,
                start_line: row.get::<_, i64>(5)?,
                end_line: row.get::<_, i64>(6)?,
                text: row.get::<_, String>(7)?,
                score: row.get::<_, f64>(8)?,
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
        params_vec.push(Value::Integer(limit as i64));

        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(params_from_iter(params_vec))?;
        let mut results = Vec::new();
        while let Some(row) = rows.next()? {
            let path: String = row.get::<_, String>(0)?;
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
            let path: String = row.get::<_, String>(0)?;
            let weight: f64 = row.get::<_, f64>(1)?;
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
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        Ok(count.unwrap_or(0))
    }

    fn update_file_dependency_edges(&self, file_path: &str) -> Result<()> {
        let has_refs: Option<i64> = self
            .conn
            .query_row(
                "SELECT 1 FROM symbol_references WHERE file_path = ? LIMIT 1",
                params![file_path],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        let has_defs: Option<i64> = self
            .conn
            .query_row(
                "SELECT 1 FROM symbols WHERE file_path = ? AND kind = 'definition' LIMIT 1",
                params![file_path],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        if has_refs.is_none() && has_defs.is_none() {
            return Ok(());
        }

        let sql_src = "WITH edges AS ( \
        SELECT r.file_path AS src_path, s.file_path AS dst_path \
          FROM symbol_references r \
          JOIN symbols s ON r.name = s.name AND s.kind = 'definition' \
         WHERE r.file_path = ? AND r.file_path <> s.file_path \
      ) \
      INSERT INTO file_dependency_edges (src_path, dst_path, reference_count) \
      SELECT src_path, dst_path, COUNT(*) AS reference_count \
        FROM edges \
       GROUP BY src_path, dst_path";

        let sql_dst = "WITH defs AS ( \
        SELECT DISTINCT name \
          FROM symbols \
         WHERE kind = 'definition' AND file_path = ? \
      ), edges AS ( \
        SELECT r.file_path AS src_path, ? AS dst_path \
          FROM symbol_references r \
          JOIN defs d ON r.name = d.name \
         WHERE r.file_path <> ? \
      ) \
      INSERT INTO file_dependency_edges (src_path, dst_path, reference_count) \
      SELECT src_path, dst_path, COUNT(*) AS reference_count \
        FROM edges \
       GROUP BY src_path, dst_path";

        self.with_transaction(|db| {
            db.conn.execute(
                "DELETE FROM file_dependency_edges WHERE src_path = ? OR dst_path = ?",
                params![file_path, file_path],
            )?;
            db.conn.execute(sql_src, params![file_path])?;
            db.conn.execute(sql_dst, params![file_path, file_path, file_path])?;
            Ok(())
        })?;
        Ok(())
    }

    fn file_dependency_pagerank(&self, limit: usize) -> Result<Vec<(String, f64)>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut stmt = self
            .conn
            .prepare("SELECT src_path, dst_path, reference_count FROM file_dependency_edges")?;
        let mut rows = stmt.query([])?;
        let mut edges = Vec::new();
        let mut nodes = BTreeMap::new();
        while let Some(row) = rows.next()? {
            let src: String = row.get::<_, String>(0)?;
            let dst: String = row.get::<_, String>(1)?;
            let weight: i64 = row.get::<_, i64>(2)?;
            let next_src = nodes.len();
            nodes.entry(src.clone()).or_insert(next_src);
            let next_dst = nodes.len();
            nodes.entry(dst.clone()).or_insert(next_dst);
            edges.push((src, dst, weight.max(0) as f64));
        }

        if nodes.is_empty() {
            return Ok(Vec::new());
        }

        let n = nodes.len();
        let mut index_to_path = vec![String::new(); n];
        for (path, idx) in &nodes {
            index_to_path[*idx] = path.clone();
        }

        let mut out_weight = vec![0.0; n];
        let mut edge_idx = Vec::with_capacity(edges.len());
        for (src, dst, weight) in edges {
            let src_idx = nodes[&src];
            let dst_idx = nodes[&dst];
            out_weight[src_idx] += weight;
            edge_idx.push((src_idx, dst_idx, weight));
        }

        let damping = 0.85;
        let mut ranks = vec![1.0 / n as f64; n];
        let mut next = vec![0.0; n];
        for _ in 0..20 {
            for val in &mut next {
                *val = (1.0 - damping) / n as f64;
            }
            let mut dangling = 0.0;
            for idx in 0..n {
                if out_weight[idx] == 0.0 {
                    dangling += ranks[idx];
                }
            }
            let dangling_share = damping * dangling / n as f64;
            for val in &mut next {
                *val += dangling_share;
            }
            for (src_idx, dst_idx, weight) in &edge_idx {
                if out_weight[*src_idx] > 0.0 {
                    next[*dst_idx] += damping * ranks[*src_idx] * (*weight / out_weight[*src_idx]);
                }
            }
            ranks.clone_from_slice(&next);
        }

        let mut scored: Vec<(String, f64)> = index_to_path
            .into_iter()
            .zip(ranks.into_iter())
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if scored.len() > limit {
            scored.truncate(limit);
        }
        Ok(scored)
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
                let chunk_hash: Vec<u8> = row.get::<_, Vec<u8>>(4)?;
                if chunk_hash.len() != 32 {
                    return Err(eyre!(
                        "invalid chunk_hash length for {}",
                        row.get::<_, String>(1)?
                    ));
                }
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&chunk_hash);
                results.push(SearchRow {
                    id: row.get::<_, String>(0)?,
                    file_path: row.get::<_, String>(1)?,
                    start_byte: row.get::<_, i64>(2)?,
                    end_byte: row.get::<_, i64>(3)?,
                    chunk_hash: hash,
                    start_line: row.get::<_, i64>(5)?,
                    end_line: row.get::<_, i64>(6)?,
                    text: row.get::<_, String>(7)?,
                    distance: row.get::<_, f64>(8)?,
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
                id: row.get::<_, String>(0)?,
                file_path: row.get::<_, String>(1)?,
                name: row.get::<_, String>(2)?,
                kind: row.get::<_, String>(3)?,
                start_byte: row.get::<_, i64>(4)? as usize,
                end_byte: row.get::<_, i64>(5)? as usize,
                language: row.get::<_, String>(6)?,
            });
        }
        Ok(symbols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use tempfile::TempDir;

    #[test]
    fn db_enforces_embedding_dim() -> Result<()> {
        let dir = TempDir::new()?;
        let db_path = dir.path().join("context.db");

        let _db = Db::open(&db_path, Some(2))?;
        let mismatch = Db::open(&db_path, Some(3));
        assert!(mismatch.is_err(), "expected embedding_dim mismatch error");
        Ok(())
    }

    #[test]
    fn db_rejects_chunk_without_file_row() -> Result<()> {
        let dir = TempDir::new()?;
        let db_path = dir.path().join("context.db");
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
            fts_text: build_fts_text("hello"),
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
        let db_path = dir.path().join("context.db");
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
            fts_text: build_fts_text("broken"),
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
        let db_path = dir.path().join("context.db");
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
        let db_path = dir.path().join("context.db");
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
        let db_path = dir.path().join("context.db");
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
        let db_path = dir.path().join("context.db");
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
                row.get::<_, i64>(0)
            })?;

        assert_eq!(
            edge_count, 0,
            "expected reference_symbol_edges to be removed when symbols are deleted"
        );
        Ok(())
    }
}
