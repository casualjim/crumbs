use std::collections::{HashMap, HashSet};
use std::path::Path;

use eyre::Result;
use gix::Pathspec;
use gix::object::tree::diff::ChangeDetached;
use gix::revision::walk::Sorting;
use regex::Regex;
use tracing::{info, warn};

use crate::repository::Repository;

pub struct HistoryConfig {
    pub depth: u32,
    pub commit_size_limit_ratio: f32,
    pub multi_parents: bool,
    pub issue_regex: String,
    pub commit_exclude_regex: Option<String>,
    pub author_exclude_regex: Option<String>,
    pub path_specs: Vec<String>,
}

pub(crate) async fn index_history(
    db: &dyn Repository,
    repo_path: &Path,
    config: &HistoryConfig,
) -> Result<()> {
    if !repo_path.join(".git").exists() {
        warn!(
            "history indexing skipped; no git repository at {}",
            repo_path.display()
        );
        return Ok(());
    }

    let known_files = db.list_files().await?;
    if known_files.is_empty() {
        return Ok(());
    }
    let known_set: HashSet<String> = known_files.into_iter().collect();

    let repo = gix::discover(repo_path)?;
    let mut head = repo.head()?;
    let Some(head_id) = head.try_peel_to_id()? else {
        return Ok(());
    };

    let commit_exclude = config
        .commit_exclude_regex
        .as_deref()
        .map(Regex::new)
        .transpose()?;
    let author_exclude = config
        .author_exclude_regex
        .as_deref()
        .map(Regex::new)
        .transpose()?;
    let issue_regex = Regex::new(&config.issue_regex)?;
    let mut pathspec = if config.path_specs.is_empty() {
        None
    } else {
        Some(Pathspec::new(
            &repo,
            false,
            config.path_specs.iter().map(|spec| spec.as_str()),
            false,
            || {
                Err(Box::new(std::io::Error::other(
                    "pathspec attribute matching is unsupported for history indexing",
                )))
            },
        )?)
    };

    let file_count = known_set.len().max(1) as f32;
    let max_files_per_commit = if config.commit_size_limit_ratio >= 1.0 {
        usize::MAX
    } else {
        (file_count * config.commit_size_limit_ratio).ceil() as usize
    }
    .max(1);

    let mut file_commit_edges: HashSet<(String, String)> = HashSet::new();
    let mut commit_issue_edges: HashSet<(String, String)> = HashSet::new();
    let mut cochange_map: HashMap<(String, String), (u64, f64)> = HashMap::new();

    let walk = repo
        .rev_walk([head_id.detach()])
        .sorting(Sorting::ByCommitTime(Default::default()));
    let walk = if config.multi_parents {
        walk
    } else {
        walk.first_parent_only()
    };
    let walk = walk.all()?;
    let mut commit_count = 0usize;

    for info in walk {
        let info = info?;
        let commit_id = info.id.to_string();
        let commit = info.object()?;
        let parent_ids: Vec<_> = commit.parent_ids().collect();

        let message = commit.message_raw()?;
        let message = std::str::from_utf8(message.as_ref()).unwrap_or("");
        if let Some(commit_exclude) = &commit_exclude
            && commit_exclude.is_match(message)
        {
            continue;
        }

        if let Some(author_exclude) = &author_exclude {
            let author = commit.author()?;
            let author_name = String::from_utf8_lossy(author.name.as_ref());
            let author_email = String::from_utf8_lossy(author.email.as_ref());
            let author_str = format!("{author_name} <{author_email}>");
            if author_exclude.is_match(&author_str) {
                continue;
            }
        }

        let issues: Vec<String> = issue_regex
            .find_iter(message)
            .map(|mat| mat.as_str().to_string())
            .collect();

        let parent_id = match parent_ids.first() {
            Some(parent_id) => parent_id,
            None => continue,
        };

        let tree = commit.tree()?;
        let parent = parent_id.object()?.try_into_commit()?;
        let parent_tree = parent.tree()?;

        let diff_options = gix::diff::Options::default();
        let changes =
            repo.diff_tree_to_tree(Some(&parent_tree), Some(&tree), Some(diff_options))?;
        let mut files = HashSet::new();
        for change in changes {
            match change {
                ChangeDetached::Addition { location, .. } => {
                    let path = String::from_utf8_lossy(location.as_ref()).into_owned();
                    if let Some(spec) = pathspec.as_mut()
                        && !spec.is_included(path.as_bytes(), None)
                    {
                        continue;
                    }
                    files.insert(path);
                }
                ChangeDetached::Modification {
                    location,
                    previous_id,
                    id,
                    previous_entry_mode,
                    entry_mode,
                    ..
                } => {
                    if previous_id == id && previous_entry_mode != entry_mode {
                        continue;
                    }
                    let path = String::from_utf8_lossy(location.as_ref()).into_owned();
                    if let Some(spec) = pathspec.as_mut()
                        && !spec.is_included(path.as_bytes(), None)
                    {
                        continue;
                    }
                    files.insert(path);
                }
                ChangeDetached::Rewrite { location, .. } => {
                    let path = String::from_utf8_lossy(location.as_ref()).into_owned();
                    if let Some(spec) = pathspec.as_mut() {
                        if spec.is_included(path.as_bytes(), None) {
                            files.insert(path);
                        }
                    } else {
                        files.insert(path);
                    }
                }
                ChangeDetached::Deletion { .. } => {}
            }
        }

        let total_files = files.len();
        if total_files == 0 || total_files > max_files_per_commit {
            continue;
        }

        let weight = 1.0 / total_files as f64;
        let mut filtered: Vec<String> = files
            .into_iter()
            .filter(|file| known_set.contains(file))
            .collect();
        if filtered.is_empty() {
            continue;
        }

        filtered.sort();
        filtered.dedup();

        for file in &filtered {
            file_commit_edges.insert((file.clone(), commit_id.clone()));
        }
        for issue in &issues {
            commit_issue_edges.insert((commit_id.clone(), issue.clone()));
        }

        if filtered.len() >= 2 {
            for i in 0..filtered.len() {
                for j in (i + 1)..filtered.len() {
                    let (left, right) = (&filtered[i], &filtered[j]);
                    let key = if left <= right {
                        (left.clone(), right.clone())
                    } else {
                        (right.clone(), left.clone())
                    };
                    let entry = cochange_map.entry(key).or_insert((0, 0.0));
                    entry.0 += 1;
                    entry.1 += weight;
                }
            }
        }

        commit_count = commit_count.saturating_add(1);
        if commit_count > config.depth as usize {
            break;
        }
    }

    let mut commit_edges: Vec<(String, String)> = file_commit_edges.into_iter().collect();
    commit_edges.sort();

    let mut commit_issue_edges: Vec<(String, String)> = commit_issue_edges.into_iter().collect();
    commit_issue_edges.sort();

    let mut cochange_edges: Vec<(String, String, i64, f64)> = cochange_map
        .into_iter()
        .map(|((src, dst), (count, weight))| (src, dst, count as i64, weight))
        .collect();
    cochange_edges.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    db.upsert_history_edges(&commit_edges, &cochange_edges)
        .await?;
    db.upsert_commit_issue_edges(&commit_issue_edges).await?;
    info!(
        "history indexing complete: commits={}, cochanges={}",
        commit_edges.len(),
        cochange_edges.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use std::process::Command;

    use super::{HistoryConfig, index_history};
    use tempfile::TempDir;
    use text_chunking::Tokenizer;

    use crate::db::Db;
    use crate::indexer::{Indexer, IndexerConfig};
    use crate::repository::Repository;
    use crate::test_support::{load_test_embedder, write_fixture_repo};

    fn run_git(repo: &Path, args: &[&str]) -> eyre::Result<String> {
        let output = Command::new("git").args(args).current_dir(repo).output()?;
        if !output.status.success() {
            return Err(eyre::eyre!(
                "git {:?} failed: {}",
                args,
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    fn write_file(root: &Path, rel: &str, contents: &str) -> eyre::Result<()> {
        let path = root.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, contents)?;
        Ok(())
    }

    #[tokio::test]
    async fn graph_build_populates_symbols_and_references() -> eyre::Result<()> {
        let (embedder, embedding_dim) = load_test_embedder()?;
        let dir = TempDir::new()?;
        write_fixture_repo(dir.path())?;

        let db_path = dir.path().join("crumbs.db");
        let db = Db::open(&db_path, Some(embedding_dim)).await?;

        let config = IndexerConfig {
            repo_path: dir.path().to_path_buf(),
            max_chunk_size: 1500,
            overlap_percentage: 0.2,
            tokenizer: Tokenizer::Tiktoken("cl100k_base".to_string()),
            max_parallel: 4,
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
        let indexer = Indexer::new(&db, embedder, config);
        indexer.index().await?;

        let test_db = libsql::Builder::new_local(&db_path).build().await?;
        let conn = test_db.connect()?;
        let mut rows = conn.query("SELECT COUNT(*) FROM symbols", ()).await?;
        let row = rows
            .next()
            .await?
            .ok_or_else(|| eyre::eyre!("missing row"))?;
        let symbols: i64 = row.get(0)?;
        let mut rows = conn
            .query("SELECT COUNT(*) FROM symbol_references", ())
            .await?;
        let row = rows
            .next()
            .await?
            .ok_or_else(|| eyre::eyre!("missing row"))?;
        let references: i64 = row.get(0)?;

        assert!(symbols > 0, "expected symbols to be populated");
        assert!(references > 0, "expected references to be populated");
        Ok(())
    }

    #[tokio::test]
    async fn history_index_matches_cupido_semantics() -> eyre::Result<()> {
        let dir = TempDir::new()?;
        let repo_path = dir.path();

        run_git(repo_path, &["init"])?;
        run_git(repo_path, &["config", "user.name", "Test User"])?;
        run_git(repo_path, &["config", "user.email", "test@example.com"])?;
        let main_branch = run_git(repo_path, &["symbolic-ref", "--short", "HEAD"])?;

        write_file(repo_path, "src/a.txt", "one\n")?;
        run_git(repo_path, &["add", "."])?;
        run_git(repo_path, &["commit", "-m", "init"])?;
        let init_id = run_git(repo_path, &["rev-parse", "HEAD"])?;

        write_file(repo_path, "src/a.txt", "two\n")?;
        run_git(repo_path, &["add", "src/a.txt"])?;
        run_git(repo_path, &["commit", "-m", "main change #123"])?;
        let main_change_id = run_git(repo_path, &["rev-parse", "HEAD"])?;

        run_git(repo_path, &["checkout", "-b", "feature"])?;
        write_file(repo_path, "src/feature.txt", "feature\n")?;
        run_git(repo_path, &["add", "src/feature.txt"])?;
        run_git(repo_path, &["commit", "-m", "feature add #999"])?;
        let feature_id = run_git(repo_path, &["rev-parse", "HEAD"])?;

        run_git(repo_path, &["checkout", main_branch.as_str()])?;
        write_file(repo_path, "docs/readme.md", "docs\n")?;
        run_git(repo_path, &["add", "docs/readme.md"])?;
        run_git(repo_path, &["commit", "-m", "docs add"])?;

        write_file(repo_path, "src/mode.sh", "#!/bin/sh\necho hi\n")?;
        run_git(repo_path, &["add", "src/mode.sh"])?;
        run_git(repo_path, &["commit", "-m", "add script"])?;
        let script_id = run_git(repo_path, &["rev-parse", "HEAD"])?;

        run_git(repo_path, &["update-index", "--chmod=+x", "src/mode.sh"])?;
        run_git(repo_path, &["commit", "-m", "mode change"])?;
        let mode_id = run_git(repo_path, &["rev-parse", "HEAD"])?;

        run_git(
            repo_path,
            &["merge", "--no-ff", "feature", "-m", "Merge feature"],
        )?;
        let merge_id = run_git(repo_path, &["rev-parse", "HEAD"])?;

        fs::remove_file(repo_path.join("src/feature.txt"))?;
        run_git(repo_path, &["add", "-A"])?;
        run_git(repo_path, &["commit", "-m", "delete feature"])?;
        let delete_id = run_git(repo_path, &["rev-parse", "HEAD"])?;

        let db_path = repo_path.join("crumbs.db");
        let db = Db::open(&db_path, Some(8)).await?;
        for file in [
            "src/a.txt",
            "src/feature.txt",
            "src/mode.sh",
            "docs/readme.md",
        ] {
            db.ensure_file_row(file, 1, None).await?;
        }

        let history = HistoryConfig {
            depth: 10240,
            commit_size_limit_ratio: 1.0,
            multi_parents: false,
            issue_regex: "(#\\d+)".to_string(),
            commit_exclude_regex: None,
            author_exclude_regex: None,
            path_specs: vec!["src/*".to_string()],
        };
        index_history(&db, repo_path, &history).await?;

        let test_db = libsql::Builder::new_local(&db_path).build().await?;
        let conn = test_db.connect()?;
        let mut rows = conn
            .query("SELECT file_path, commit_id FROM file_commit_edges", ())
            .await?;
        let mut edges: HashMap<String, Vec<String>> = HashMap::new();
        while let Some(row) = rows.next().await? {
            let file: String = row.get(0)?;
            let commit: String = row.get(1)?;
            edges.entry(file).or_default().push(commit);
        }
        for commits in edges.values_mut() {
            commits.sort();
        }

        let mut rows = conn
            .query("SELECT commit_id, issue_id FROM commit_issue_edges", ())
            .await?;
        let mut issue_edges: HashMap<String, Vec<String>> = HashMap::new();
        while let Some(row) = rows.next().await? {
            let commit: String = row.get(0)?;
            let issue: String = row.get(1)?;
            issue_edges.entry(commit).or_default().push(issue);
        }
        for issues in issue_edges.values_mut() {
            issues.sort();
        }

        let a_commits = edges.get("src/a.txt").cloned().unwrap_or_default();
        assert!(a_commits.contains(&main_change_id));
        assert!(!a_commits.contains(&init_id));

        let feature_commits = edges.get("src/feature.txt").cloned().unwrap_or_default();
        assert_eq!(feature_commits, vec![merge_id.clone()]);
        assert!(!feature_commits.contains(&feature_id));
        assert!(!feature_commits.contains(&delete_id));

        let docs_commits = edges.get("docs/readme.md").cloned().unwrap_or_default();
        assert!(docs_commits.is_empty());

        let mode_commits = edges.get("src/mode.sh").cloned().unwrap_or_default();
        assert!(mode_commits.contains(&script_id));
        assert!(!mode_commits.contains(&mode_id));

        let main_issues = issue_edges
            .get(&main_change_id)
            .cloned()
            .unwrap_or_default();
        assert_eq!(main_issues, vec!["#123".to_string()]);
        assert!(!issue_edges.contains_key(&feature_id));

        Ok(())
    }
}
