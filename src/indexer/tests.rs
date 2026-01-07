use super::*;
use std::thread::sleep;
use std::time::Duration;

use rusqlite::params;
use rusqlite::types::Value;
use tempfile::TempDir;
use text_chunking::Tokenizer;

use crate::db::{ChunkRecord, GraphData, ReferenceRecord, SymbolRecord};
use crate::graph::HistoryConfig;
use crate::repository::Repository;
use crate::search;
use crate::test_support::{load_test_embedder, write_fixture_repo};
use crate::Db;

fn make_hash(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn setup_db() -> (Db, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let db_path = dir.path().join("context.db");
    let db = Db::open(&db_path, Some(2)).expect("db open");
    (db, dir)
}

fn chunk_record(
    id: &str,
    file_path: &str,
    chunk_hash: [u8; 32],
    tokens: Option<Vec<u32>>,
) -> ChunkRecord {
    ChunkRecord {
        id: id.to_string(),
        file_path: file_path.to_string(),
        start_byte: 0,
        end_byte: 10,
        chunk_hash,
        start_line: 1,
        end_line: 1,
        text: "hello world".to_string(),
        kind: "text".to_string(),
        ordinal: 0,
        tokens,
    }
}

fn query_updated_at(dir: &TempDir, file_path: &str) -> String {
    let db_path = dir.path().join("context.db");
    let conn = rusqlite::Connection::open(db_path).expect("open conn");
    conn.query_row(
        "SELECT CAST(updated_at AS VARCHAR) FROM files WHERE path = ?",
        params![file_path],
        |row| row.get::<_, String>(0),
    )
    .expect("updated_at")
}

fn query_tokens(dir: &TempDir, chunk_id: &str) -> Vec<i32> {
    let db_path = dir.path().join("context.db");
    let conn = rusqlite::Connection::open(db_path).expect("open conn");
    let value: Value = conn
        .query_row("SELECT tokens FROM chunks WHERE id = ?", params![chunk_id], |row| {
            row.get::<_, Value>(0)
        })
        .expect("tokens");
    match value {
        Value::Blob(blob) => decode_tokens(&blob),
        Value::Null => Vec::new(),
        other => panic!("unexpected tokens value: {other:?}"),
    }
}

fn decode_tokens(blob: &[u8]) -> Vec<i32> {
    if blob.is_empty() {
        return Vec::new();
    }
    if blob.len() % 4 != 0 {
        panic!("invalid token blob length {}", blob.len());
    }
    blob.chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn insert_files(db: &Db, paths: &[&str]) {
    for path in paths {
        db.upsert_file_metadata(path, 0, make_hash(1), None)
            .expect("insert file");
    }
}

#[tokio::test]
async fn end_to_end_index_and_search() -> Result<()> {
    let (embedder, embedding_dim) = load_test_embedder()?;
    let dir = TempDir::new()?;
    write_fixture_repo(dir.path())?;

    let db_path = dir.path().join("context.db");
    let db = Db::open(&db_path, Some(embedding_dim))?;
    let tokenizer = Tokenizer::Tiktoken("cl100k_base".to_string());
    let config = IndexerConfig {
        repo_path: dir.path().to_path_buf(),
        max_chunk_size: 512,
        overlap_percentage: 0.1,
        tokenizer: tokenizer.clone(),
        max_parallel: 2,
        max_file_size: Some(5 * 1024 * 1024),
        large_file_threads: 2,
        max_batch_size: 16,
        max_tokens: 8_192,
        embedding_workers: 1,
        cancel_token: None,
        history: HistoryConfig {
            depth: 10240,
            commit_size_limit_ratio: 1.0,
            multi_parents: false,
            issue_regex: "(#\\d+)".to_string(),
            commit_exclude_regex: None,
            author_exclude_regex: None,
            path_specs: Vec::new(),
        },
    };
    let indexer = Indexer::new(&db, embedder.clone(), config);
    indexer.index().await?;

    let db = Db::open(&db_path, Some(embedding_dim))?;
    let mut search_config = search::SearchConfig::new(5, 0.6);
    search_config.min_score = 0.0;
    search_config.rerank = false;
    let results = search::search(
        &db,
        &embedder,
        None,
        &tokenizer,
        "add numbers",
        search_config,
    )
    .await?;

    assert!(!results.is_empty(), "expected search to return results");
    Ok(())
}

#[test]
fn unchanged_file_chunks_should_not_bump_updated_at() {
    let (db, dir) = setup_db();
    let hash = make_hash(1);
    db.upsert_file_metadata("a.rs", 10, hash, None)
        .expect("insert");

    let before = query_updated_at(&dir, "a.rs");
    sleep(Duration::from_millis(5));
    db.upsert_file_metadata("a.rs", 10, hash, None)
        .expect("replace");
    let after = query_updated_at(&dir, "a.rs");

    assert_eq!(before, after, "updated_at should not change for identical chunks");
}

#[test]
fn unchanged_file_graph_should_not_bump_updated_at() {
    let (db, dir) = setup_db();
    let hash = make_hash(2);
    let graph = GraphData {
        symbols: vec![],
        references: vec![],
        resolutions: vec![],
    };
    db.upsert_file_graph("a.rs", 10, hash, "rust", None, graph.clone())
        .expect("graph insert");

    let before = query_updated_at(&dir, "a.rs");
    sleep(Duration::from_secs(1));
    db.upsert_file_graph("a.rs", 10, hash, "rust", None, graph)
        .expect("graph replace");
    let after = query_updated_at(&dir, "a.rs");

    assert_eq!(before, after, "updated_at should not change for identical graph");
}

#[test]
fn replace_history_edges_should_preserve_existing_edges() {
    let (db, dir) = setup_db();
    insert_files(&db, &["a.rs", "b.rs"]);
    let db_path = dir.path().join("context.db");
    let conn = rusqlite::Connection::open(db_path.clone()).expect("open conn");
    conn.execute(
        "INSERT INTO file_commit_edges (file_path, commit_id) VALUES (?, ?)",
        params!["a.rs", "c1"],
    )
    .expect("insert edge");
    conn.execute(
        "INSERT INTO file_cochange_edges (src_path, dst_path, commit_count, weight) VALUES (?, ?, ?, ?)",
        params!["a.rs", "b.rs", 1i64, 1.0f64],
    )
    .expect("insert cochange");

    db.upsert_history_edges(&[("b.rs".to_string(), "c2".to_string())], &[])
        .expect("upsert history");

    let conn = rusqlite::Connection::open(db_path).expect("open conn");
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM file_commit_edges WHERE file_path = 'a.rs'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .expect("count");
    assert!(
        count > 0,
        "history edges should remain until explicitly pruned"
    );
}

#[test]
fn refresh_file_dependency_edges_should_preserve_existing_edges() {
    let (db, dir) = setup_db();
    insert_files(&db, &["a.rs", "b.rs"]);
    let db_path = dir.path().join("context.db");
    let conn = rusqlite::Connection::open(db_path.clone()).expect("open conn");
    conn.execute(
        "INSERT INTO file_dependency_edges (src_path, dst_path, reference_count) VALUES (?, ?, ?)",
        params!["a.rs", "b.rs", 1i64],
    )
    .expect("insert dependency");

    db.update_file_dependency_edges("a.rs")
        .expect("update deps");

    let conn = rusqlite::Connection::open(db_path).expect("open conn");
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM file_dependency_edges WHERE src_path = 'a.rs'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .expect("count");
    assert!(count > 0, "existing dependency edges should be preserved");
}

#[test]
fn find_chunk_id_should_dedupe_across_files() {
    let (db, _dir) = setup_db();
    let hash = make_hash(3);
    let record = chunk_record("c1", "a.rs", hash, None);
    db.upsert_file_metadata("a.rs", 10, hash, None)
        .expect("insert file");
    db.upsert_chunk_with_embedding(&record, &[0.1, 0.2])
        .expect("insert chunk");

    let found = db
        .find_chunk_id("b.rs", 0, 10, "text", hash)
        .expect("find");
    assert!(found.is_some(), "should reuse embedding across files");
}

#[test]
fn find_chunk_id_should_dedupe_when_offsets_shift() {
    let (db, _dir) = setup_db();
    let hash = make_hash(4);
    let record = chunk_record("c1", "a.rs", hash, None);
    db.upsert_file_metadata("a.rs", 10, hash, None)
        .expect("insert file");
    db.upsert_chunk_with_embedding(&record, &[0.1, 0.2])
        .expect("insert chunk");

    let found = db
        .find_chunk_id("a.rs", 5, 15, "text", hash)
        .expect("find");
    assert!(found.is_some(), "should reuse embedding when offsets shift");
}

#[test]
fn tokens_should_update_without_embedding() {
    let (db, dir) = setup_db();
    let hash = make_hash(5);
    let record = chunk_record("c1", "a.rs", hash, Some(vec![1, 2]));
    db.upsert_file_metadata("a.rs", 10, hash, None)
        .expect("insert file");
    db.upsert_chunk_with_embedding(&record, &[0.1, 0.2])
        .expect("insert chunk");

    let updated = ChunkRecord {
        tokens: Some(vec![9, 9]),
        ..record
    };
    db.update_chunk_without_embedding(&updated)
        .expect("update");

    let tokens = query_tokens(&dir, "c1");
    assert_eq!(tokens, vec![9, 9], "tokens should be updated for existing chunks");
}

#[test]
fn replace_file_graph_should_preserve_unmentioned_symbols() {
    let (db, dir) = setup_db();
    insert_files(&db, &["a.rs"]);
    let hash = make_hash(6);
    let graph = GraphData {
        symbols: vec![SymbolRecord {
            id: "s1".to_string(),
            file_path: "a.rs".to_string(),
            name: "foo".to_string(),
            kind: "definition".to_string(),
            start_byte: 0,
            end_byte: 1,
            language: "rust".to_string(),
        }],
        references: vec![],
        resolutions: vec![],
    };
    db.upsert_file_graph("a.rs", 10, hash, "rust", None, graph)
        .expect("graph insert");

    let empty = GraphData {
        symbols: vec![],
        references: vec![],
        resolutions: vec![],
    };
    db.upsert_file_graph("a.rs", 10, hash, "rust", None, empty)
        .expect("graph replace");

    let db_path = dir.path().join("context.db");
    let conn = rusqlite::Connection::open(db_path).expect("open conn");
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM symbols WHERE file_path = 'a.rs'", [], |row| {
            row.get::<_, i64>(0)
        })
        .expect("count");
    assert_eq!(count, 0, "removed symbols should be deleted");
}

#[test]
fn replace_file_graph_should_preserve_unmentioned_references() {
    let (db, dir) = setup_db();
    insert_files(&db, &["a.rs"]);
    let hash = make_hash(7);
    let graph = GraphData {
        symbols: vec![],
        references: vec![ReferenceRecord {
            id: "r1".to_string(),
            file_path: "a.rs".to_string(),
            name: "foo".to_string(),
            start_byte: 0,
            end_byte: 1,
            language: "rust".to_string(),
        }],
        resolutions: vec![],
    };
    db.upsert_file_graph("a.rs", 10, hash, "rust", None, graph)
        .expect("graph insert");

    let empty = GraphData {
        symbols: vec![],
        references: vec![],
        resolutions: vec![],
    };
    db.upsert_file_graph("a.rs", 10, hash, "rust", None, empty)
        .expect("graph replace");

    let db_path = dir.path().join("context.db");
    let conn = rusqlite::Connection::open(db_path).expect("open conn");
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM symbol_references WHERE file_path = 'a.rs'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .expect("count");
    assert_eq!(count, 0, "removed references should be deleted");
}

#[test]
fn refresh_file_dependency_edges_should_include_multi_definitions() {
    let (db, dir) = setup_db();
    insert_files(&db, &["a.rs", "b.rs", "c.rs"]);
    let graph_a = GraphData {
        symbols: vec![SymbolRecord {
            id: "s1".to_string(),
            file_path: String::new(),
            name: "X".to_string(),
            kind: "definition".to_string(),
            start_byte: 0,
            end_byte: 1,
            language: "rust".to_string(),
        }],
        references: Vec::new(),
        resolutions: Vec::new(),
    };
    db.upsert_file_graph("a.rs", 1, make_hash(10), "rust", None, graph_a)
        .expect("graph a");

    let graph_b = GraphData {
        symbols: vec![SymbolRecord {
            id: "s2".to_string(),
            file_path: String::new(),
            name: "X".to_string(),
            kind: "definition".to_string(),
            start_byte: 0,
            end_byte: 1,
            language: "rust".to_string(),
        }],
        references: Vec::new(),
        resolutions: Vec::new(),
    };
    db.upsert_file_graph("b.rs", 1, make_hash(11), "rust", None, graph_b)
        .expect("graph b");

    let graph_c = GraphData {
        symbols: Vec::new(),
        references: vec![ReferenceRecord {
            id: "r1".to_string(),
            file_path: String::new(),
            name: "X".to_string(),
            start_byte: 0,
            end_byte: 1,
            language: "rust".to_string(),
        }],
        resolutions: Vec::new(),
    };
    db.upsert_file_graph("c.rs", 1, make_hash(12), "rust", None, graph_c)
        .expect("graph c");

    db.update_file_dependency_edges("c.rs")
        .expect("update deps");

    let db_path = dir.path().join("context.db");
    let conn = rusqlite::Connection::open(&db_path).expect("open conn");
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM file_dependency_edges WHERE src_path = 'c.rs'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .expect("count");
    assert!(count > 0, "multi-definition symbols should still create dependencies");
}
