use super::*;
use std::time::Duration;

use libsql::{Builder, Value, params};
use tempfile::TempDir;
use text_chunking::Tokenizer;
use tokio::time::sleep;

use crate::db::{ChunkRecord, GraphData, ReferenceRecord, SymbolRecord, build_fts_text};
use crate::graph::HistoryConfig;
use crate::repository::Repository;
use crate::search;
use crate::test_support::{load_test_embedder, load_test_reranker, write_fixture_repo};
use crate::Db;

fn make_hash(byte: u8) -> [u8; 32] {
    [byte; 32]
}

async fn setup_db() -> (Db, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let db_path = dir.path().join("crumbs.db");
    let db = Db::open(&db_path, Some(2))
        .await
        .expect("db open");
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
        fts_text: build_fts_text("hello world"),
        kind: "text".to_string(),
        ordinal: 0,
        tokens,
    }
}

async fn query_updated_at(dir: &TempDir, file_path: &str) -> String {
    let db_path = dir.path().join("crumbs.db");
    let db = Builder::new_local(db_path).build().await.expect("open db");
    let conn = db.connect().expect("open conn");
    let mut rows = conn
        .query(
            "SELECT CAST(updated_at AS TEXT) FROM files WHERE path = ?",
            params![file_path],
        )
        .await
        .expect("query");
    let row = rows.next().await.expect("rows").expect("row");
    row.get::<String>(0).expect("updated_at")
}

async fn query_tokens(dir: &TempDir, chunk_id: &str) -> Vec<i32> {
    let db_path = dir.path().join("crumbs.db");
    let db = Builder::new_local(db_path).build().await.expect("open db");
    let conn = db.connect().expect("open conn");
    let mut rows = conn
        .query("SELECT tokens FROM chunks WHERE id = ?", params![chunk_id])
        .await
        .expect("query tokens");
    let row = rows.next().await.expect("rows").expect("row");
    let value: Value = row.get(0).expect("tokens");
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

async fn insert_files(db: &Db, paths: &[&str]) {
    for path in paths {
        db.upsert_file_metadata(path, 0, make_hash(1), None)
            .await
            .expect("insert file");
    }
}

#[tokio::test]
async fn end_to_end_index_and_search() -> Result<()> {
    let (embedder, embedding_dim) = load_test_embedder()?;
    let dir = TempDir::new()?;
    write_fixture_repo(dir.path())?;

    let db_path = dir.path().join("crumbs.db");
    let db = Db::open(&db_path, Some(embedding_dim)).await?;
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

    let db = Db::open(&db_path, Some(embedding_dim)).await?;
    let mut search_config = search::SearchConfig::new(5, 0.6);
    search_config.min_score = 0.0;
    let reranker = load_test_reranker()?;
    let search_ctx = search::SearchContext {
        db: &db,
        embedder: &embedder,
        reranker: &reranker,
        tokenizer: &tokenizer,
        progress: None,
    };
    let results = search::search(&search_ctx, "add numbers", search_config).await?;

    assert!(!results.is_empty(), "expected search to return results");
    Ok(())
}

#[tokio::test]
async fn unchanged_file_chunks_should_not_bump_updated_at() {
    let (db, dir) = setup_db().await;
    let hash = make_hash(1);
    db.upsert_file_metadata("a.rs", 10, hash, None)
        .await
        .expect("insert");

    let before = query_updated_at(&dir, "a.rs").await;
    sleep(Duration::from_millis(5)).await;
    db.upsert_file_metadata("a.rs", 10, hash, None)
        .await
        .expect("replace");
    let after = query_updated_at(&dir, "a.rs").await;

    assert_eq!(before, after, "updated_at should not change for identical chunks");
}

#[tokio::test]
async fn unchanged_file_graph_should_not_bump_updated_at() {
    let (db, dir) = setup_db().await;
    let hash = make_hash(2);
    let graph = GraphData {
        symbols: vec![],
        references: vec![],
        resolutions: vec![],
    };
    db.upsert_file_graph("a.rs", 10, hash, "rust", None, graph.clone())
        .await
        .expect("graph insert");

    let before = query_updated_at(&dir, "a.rs").await;
    sleep(Duration::from_secs(1)).await;
    db.upsert_file_graph("a.rs", 10, hash, "rust", None, graph)
        .await
        .expect("graph replace");
    let after = query_updated_at(&dir, "a.rs").await;

    assert_eq!(before, after, "updated_at should not change for identical graph");
}

#[tokio::test]
async fn replace_history_edges_should_preserve_existing_edges() {
    let (db, dir) = setup_db().await;
    insert_files(&db, &["a.rs", "b.rs"]).await;
    let db_path = dir.path().join("crumbs.db");
    let test_db = Builder::new_local(db_path.clone()).build().await.expect("open db");
    let conn = test_db.connect().expect("open conn");
    conn.execute(
        "INSERT INTO file_commit_edges (file_path, commit_id) VALUES (?, ?)",
        params!["a.rs", "c1"],
    )
    .await
    .expect("insert edge");
    conn.execute(
        "INSERT INTO file_cochange_edges (src_path, dst_path, commit_count, weight) VALUES (?, ?, ?, ?)",
        params!["a.rs", "b.rs", 1i64, 1.0f64],
    )
    .await
    .expect("insert cochange");

    db.upsert_history_edges(&[("b.rs".to_string(), "c2".to_string())], &[])
        .await
        .expect("upsert history");

    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM file_commit_edges WHERE file_path = 'a.rs'",
            (),
        )
        .await
        .expect("count query");
    let row = rows.next().await.expect("rows").expect("row");
    let count: i64 = row.get(0).expect("count");
    assert!(
        count > 0,
        "history edges should remain until explicitly pruned"
    );
}

#[tokio::test]
async fn refresh_file_dependency_edges_should_preserve_existing_edges() {
    let (db, dir) = setup_db().await;
    insert_files(&db, &["a.rs", "b.rs"]).await;
    let db_path = dir.path().join("crumbs.db");
    let test_db = Builder::new_local(db_path.clone()).build().await.expect("open db");
    let conn = test_db.connect().expect("open conn");
    conn.execute(
        "INSERT INTO file_dependency_edges (src_path, dst_path, reference_count) VALUES (?, ?, ?)",
        params!["a.rs", "b.rs", 1i64],
    )
    .await
    .expect("insert dependency");

    db.update_file_dependency_edges("a.rs")
        .await
        .expect("update deps");

    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM file_dependency_edges WHERE src_path = 'a.rs'",
            (),
        )
        .await
        .expect("count query");
    let row = rows.next().await.expect("rows").expect("row");
    let count: i64 = row.get(0).expect("count");
    assert!(count > 0, "existing dependency edges should be preserved");
}

#[tokio::test]
async fn find_chunk_id_should_dedupe_across_files() {
    let (db, _dir) = setup_db().await;
    let hash = make_hash(3);
    let record = chunk_record("c1", "a.rs", hash, None);
    db.upsert_file_metadata("a.rs", 10, hash, None)
        .await
        .expect("insert file");
    db.upsert_chunk_with_embedding(&record, &[0.1, 0.2])
        .await
        .expect("insert chunk");

    let found = db
        .find_chunk_id("b.rs", 0, 10, "text", hash)
        .await
        .expect("find");
    assert!(found.is_some(), "should reuse embedding across files");
}

#[tokio::test]
async fn find_chunk_id_should_dedupe_when_offsets_shift() {
    let (db, _dir) = setup_db().await;
    let hash = make_hash(4);
    let record = chunk_record("c1", "a.rs", hash, None);
    db.upsert_file_metadata("a.rs", 10, hash, None)
        .await
        .expect("insert file");
    db.upsert_chunk_with_embedding(&record, &[0.1, 0.2])
        .await
        .expect("insert chunk");

    let found = db
        .find_chunk_id("a.rs", 5, 15, "text", hash)
        .await
        .expect("find");
    assert!(found.is_some(), "should reuse embedding when offsets shift");
}

#[tokio::test]
async fn tokens_should_update_without_embedding() {
    let (db, dir) = setup_db().await;
    let hash = make_hash(5);
    let record = chunk_record("c1", "a.rs", hash, Some(vec![1, 2]));
    db.upsert_file_metadata("a.rs", 10, hash, None)
        .await
        .expect("insert file");
    db.upsert_chunk_with_embedding(&record, &[0.1, 0.2])
        .await
        .expect("insert chunk");

    let updated = ChunkRecord {
        tokens: Some(vec![9, 9]),
        ..record
    };
    db.update_chunk_without_embedding(&updated)
        .await
        .expect("update");

    let tokens = query_tokens(&dir, "c1").await;
    assert_eq!(tokens, vec![9, 9], "tokens should be updated for existing chunks");
}

#[tokio::test]
async fn replace_file_graph_should_preserve_unmentioned_symbols() {
    let (db, dir) = setup_db().await;
    insert_files(&db, &["a.rs"]).await;
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
        .await
        .expect("graph insert");

    let empty = GraphData {
        symbols: vec![],
        references: vec![],
        resolutions: vec![],
    };
    db.upsert_file_graph("a.rs", 10, hash, "rust", None, empty)
        .await
        .expect("graph replace");

    let db_path = dir.path().join("crumbs.db");
    let test_db = Builder::new_local(db_path).build().await.expect("open db");
    let conn = test_db.connect().expect("open conn");
    let mut rows = conn
        .query("SELECT COUNT(*) FROM symbols WHERE file_path = 'a.rs'", ())
        .await
        .expect("count query");
    let row = rows.next().await.expect("rows").expect("row");
    let count: i64 = row.get(0).expect("count");
    assert_eq!(count, 0, "removed symbols should be deleted");
}

#[tokio::test]
async fn replace_file_graph_should_preserve_unmentioned_references() {
    let (db, dir) = setup_db().await;
    insert_files(&db, &["a.rs"]).await;
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
        .await
        .expect("graph insert");

    let empty = GraphData {
        symbols: vec![],
        references: vec![],
        resolutions: vec![],
    };
    db.upsert_file_graph("a.rs", 10, hash, "rust", None, empty)
        .await
        .expect("graph replace");

    let db_path = dir.path().join("crumbs.db");
    let test_db = Builder::new_local(db_path).build().await.expect("open db");
    let conn = test_db.connect().expect("open conn");
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM symbol_references WHERE file_path = 'a.rs'",
            (),
        )
        .await
        .expect("count query");
    let row = rows.next().await.expect("rows").expect("row");
    let count: i64 = row.get(0).expect("count");
    assert_eq!(count, 0, "removed references should be deleted");
}

#[tokio::test]
async fn refresh_file_dependency_edges_should_include_multi_definitions() {
    let (db, dir) = setup_db().await;
    insert_files(&db, &["a.rs", "b.rs", "c.rs"]).await;
    let graph_a = GraphData {
        symbols: vec![SymbolRecord {
            id: "s1".to_string(),
            file_path: "a.rs".to_string(),
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
        .await
        .expect("graph a");

    let graph_b = GraphData {
        symbols: vec![SymbolRecord {
            id: "s2".to_string(),
            file_path: "b.rs".to_string(),
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
        .await
        .expect("graph b");

    let graph_c = GraphData {
        symbols: Vec::new(),
        references: vec![ReferenceRecord {
            id: "r1".to_string(),
            file_path: "c.rs".to_string(),
            name: "X".to_string(),
            start_byte: 0,
            end_byte: 1,
            language: "rust".to_string(),
        }],
        resolutions: Vec::new(),
    };
    db.upsert_file_graph("c.rs", 1, make_hash(12), "rust", None, graph_c)
        .await
        .expect("graph c");

    db.update_file_dependency_edges("c.rs")
        .await
        .expect("update deps");

    let db_path = dir.path().join("crumbs.db");
    let test_db = Builder::new_local(&db_path).build().await.expect("open db");
    let conn = test_db.connect().expect("open conn");
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM file_dependency_edges WHERE src_path = 'c.rs'",
            (),
        )
        .await
        .expect("count query");
    let row = rows.next().await.expect("rows").expect("row");
    let count: i64 = row.get(0).expect("count");
    assert!(count > 0, "multi-definition symbols should still create dependencies");
}
