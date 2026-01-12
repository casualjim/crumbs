use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use eyre::Result;
use jiff::ToSpan;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::db::Db;

const STATUS_TOMBSTONE: &str = "tombstone";
const STATUS_CLOSED: &str = "closed";
const DEFAULT_TOMBSTONE_TTL_DAYS: i64 = 30;
const CLOCK_SKEW_GRACE_HOURS: i64 = 1;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct IssueKey {
    id: String,
    created_at: Option<String>,
    sender: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Dependency {
    pub issue_id: String,
    pub depends_on_id: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub created_at: String,
    pub created_by: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Comment {
    pub id: String,
    pub issue_id: String,
    pub author: String,
    pub text: String,
    pub created_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Issue {
    pub id: String,
    #[serde(skip)]
    pub content_hash: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub design: String,
    #[serde(default)]
    pub acceptance_criteria: String,
    #[serde(default)]
    pub notes: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub issue_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assignee: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_minutes: Option<i32>,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub closed_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_ref: Option<String>,
    #[serde(default)]
    pub sender: String,
    #[serde(default)]
    pub ephemeral: bool,
    #[serde(default)]
    pub replies_to: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relates_to: Vec<String>,
    #[serde(default)]
    pub duplicate_of: String,
    #[serde(default)]
    pub superseded_by: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deleted_at: Option<String>,
    #[serde(default)]
    pub deleted_by: String,
    #[serde(default)]
    pub delete_reason: String,
    #[serde(default)]
    pub original_type: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dependencies: Vec<Dependency>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub comments: Vec<Comment>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub affected_symbols: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solid_volume: Option<String>,
    #[serde(default)]
    pub topology_hash: String,
    #[serde(default)]
    pub is_solid: bool,
}

impl Issue {
    pub fn new(
        id: String,
        title: String,
        description: String,
        issue_type: String,
        priority: i32,
    ) -> Self {
        let now = now_timestamp();
        Self {
            id,
            content_hash: String::new(),
            title,
            description,
            design: String::new(),
            acceptance_criteria: String::new(),
            notes: String::new(),
            status: "open".to_string(),
            priority,
            issue_type,
            assignee: None,
            estimated_minutes: None,
            created_at: now.clone(),
            updated_at: now,
            closed_at: None,
            external_ref: None,
            sender: String::new(),
            ephemeral: false,
            replies_to: String::new(),
            relates_to: Vec::new(),
            duplicate_of: String::new(),
            superseded_by: String::new(),
            deleted_at: None,
            deleted_by: String::new(),
            delete_reason: String::new(),
            original_type: String::new(),
            labels: Vec::new(),
            dependencies: Vec::new(),
            comments: Vec::new(),
            affected_symbols: Vec::new(),
            solid_volume: None,
            topology_hash: String::new(),
            is_solid: false,
        }
    }

    pub fn touch(&mut self) {
        self.updated_at = now_timestamp();
    }

    pub fn close(&mut self) {
        self.status = "closed".to_string();
        let now = now_timestamp();
        self.updated_at = now.clone();
        self.closed_at = Some(now);
    }

    pub fn summary_query(&self) -> String {
        let mut parts = Vec::new();
        if !self.title.trim().is_empty() {
            parts.push(self.title.clone());
        }
        if !self.description.trim().is_empty() {
            parts.push(self.description.clone());
        }
        if !self.design.trim().is_empty() {
            parts.push(self.design.clone());
        }
        if !self.acceptance_criteria.trim().is_empty() {
            parts.push(self.acceptance_criteria.clone());
        }
        if !self.notes.trim().is_empty() {
            parts.push(self.notes.clone());
        }
        if !self.labels.is_empty() {
            parts.push(self.labels.join(" "));
        }
        if !self.affected_symbols.is_empty() {
            parts.push(self.affected_symbols.join(" "));
        }
        parts.join("\n")
    }

    pub fn dependency_ids(&self) -> Vec<String> {
        self.dependencies
            .iter()
            .map(|dep| dep.depends_on_id.clone())
            .collect()
    }
}

#[derive(Clone, Debug, Default)]
pub struct IssueFilters {
    pub status: Option<String>,
    pub assignee: Option<String>,
    pub label: Option<String>,
    pub issue_type: Option<String>,
    pub priority: Option<i32>,
    pub limit: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct IssueSearchResult {
    pub issue: Issue,
    pub score: f64,
}

pub fn issues_jsonl_path(data_dir: &Path) -> PathBuf {
    data_dir.join("issues.jsonl")
}

pub async fn sync_from_jsonl(db: &Db, path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let issues = read_issues_jsonl(path)?;
    for issue in issues {
        db.upsert_issue(&issue).await?;
    }
    Ok(())
}

pub async fn export_to_jsonl(db: &Db, path: &Path) -> Result<()> {
    let issues = db.list_all_issues().await?;
    write_issues_jsonl(&issues, path)?;
    Ok(())
}

pub fn read_issues_jsonl(path: &Path) -> Result<Vec<Issue>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut issues = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let issue: Issue = serde_json::from_str(trimmed)?;
        issues.push(issue);
    }
    Ok(issues)
}

pub fn write_issues_jsonl(issues: &[Issue], path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut sorted: Vec<Issue> = issues.to_vec();
    sorted.sort_by(|a, b| a.id.cmp(&b.id));
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for issue in sorted {
        let line = serde_json::to_string(&issue)?;
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

pub fn merge_issues_3way(base: &[Issue], local: &[Issue], remote: &[Issue]) -> Vec<Issue> {
    let base_map = build_issue_map(base);
    let (local_map, local_by_id) = build_issue_maps(local);
    let (remote_map, remote_by_id) = build_issue_maps(remote);

    let mut all_keys: HashSet<IssueKey> = HashSet::new();
    all_keys.extend(base_map.keys().cloned());
    all_keys.extend(local_map.keys().cloned());
    all_keys.extend(remote_map.keys().cloned());

    let mut processed_keys: HashSet<IssueKey> = HashSet::new();
    let mut processed_ids: HashSet<String> = HashSet::new();
    let mut merged = Vec::new();

    for key in all_keys {
        if processed_keys.contains(&key) {
            continue;
        }
        processed_keys.insert(key.clone());

        let base_issue = base_map.get(&key).cloned();
        let mut left_issue = local_map.get(&key).cloned();
        let mut right_issue = remote_map.get(&key).cloned();

        let mut in_left = left_issue.is_some();
        let mut in_right = right_issue.is_some();

        if !in_left
            && in_right
            && let Some(right) = &right_issue
            && let Some(fallback) = local_by_id.get(&right.id)
        {
            left_issue = Some(fallback.clone());
            in_left = true;
            processed_keys.insert(issue_key(fallback));
        }
        if !in_right
            && in_left
            && let Some(left) = &left_issue
            && let Some(fallback) = remote_by_id.get(&left.id)
        {
            right_issue = Some(fallback.clone());
            in_right = true;
            processed_keys.insert(issue_key(fallback));
        }

        if processed_ids.contains(&key.id) {
            continue;
        }
        processed_ids.insert(key.id.clone());

        let in_base = base_issue.is_some();
        let left_tombstone = in_left && is_tombstone(left_issue.as_ref().unwrap());
        let right_tombstone = in_right && is_tombstone(right_issue.as_ref().unwrap());

        if in_base && in_left && in_right {
            let base_i = base_issue.unwrap();
            let left_i = left_issue.unwrap();
            let right_i = right_issue.unwrap();
            if left_tombstone && right_tombstone {
                merged.push(merge_tombstones(left_i, right_i));
            } else if left_tombstone && !right_tombstone {
                if is_expired_tombstone(&left_i, DEFAULT_TOMBSTONE_TTL_DAYS) {
                    merged.push(right_i);
                } else {
                    merged.push(left_i);
                }
            } else if right_tombstone && !left_tombstone {
                if is_expired_tombstone(&right_i, DEFAULT_TOMBSTONE_TTL_DAYS) {
                    merged.push(left_i);
                } else {
                    merged.push(right_i);
                }
            } else {
                merged.push(merge_issue(base_i, left_i, right_i));
            }
        } else if !in_base && in_left && in_right {
            let left_i = left_issue.unwrap();
            let right_i = right_issue.unwrap();
            if left_tombstone && right_tombstone {
                merged.push(merge_tombstones(left_i, right_i));
            } else if left_tombstone && !right_tombstone {
                if is_expired_tombstone(&left_i, DEFAULT_TOMBSTONE_TTL_DAYS) {
                    merged.push(right_i);
                } else {
                    merged.push(left_i);
                }
            } else if right_tombstone && !left_tombstone {
                if is_expired_tombstone(&right_i, DEFAULT_TOMBSTONE_TTL_DAYS) {
                    merged.push(left_i);
                } else {
                    merged.push(right_i);
                }
            } else {
                let empty_base = empty_issue_from(&left_i);
                merged.push(merge_issue(empty_base, left_i, right_i));
            }
        } else if in_base && in_left && !in_right {
            if left_tombstone {
                merged.push(left_issue.unwrap());
            }
        } else if in_base && !in_left && in_right {
            if right_tombstone {
                merged.push(right_issue.unwrap());
            }
        } else if !in_base && in_left && !in_right {
            merged.push(left_issue.unwrap());
        } else if !in_base && !in_left && in_right {
            merged.push(right_issue.unwrap());
        }
    }

    merged
}

fn issue_key(issue: &Issue) -> IssueKey {
    IssueKey {
        id: issue.id.clone(),
        created_at: if issue.created_at.trim().is_empty() {
            None
        } else {
            Some(issue.created_at.clone())
        },
        sender: issue.sender.clone(),
    }
}

fn build_issue_map(issues: &[Issue]) -> HashMap<IssueKey, Issue> {
    let mut map = HashMap::new();
    for issue in issues {
        map.insert(issue_key(issue), issue.clone());
    }
    map
}

fn build_issue_maps(issues: &[Issue]) -> (HashMap<IssueKey, Issue>, HashMap<String, Issue>) {
    let mut key_map = HashMap::new();
    let mut id_map = HashMap::new();
    for issue in issues {
        key_map.insert(issue_key(issue), issue.clone());
        id_map.insert(issue.id.clone(), issue.clone());
    }
    (key_map, id_map)
}

fn empty_issue_from(issue: &Issue) -> Issue {
    Issue {
        id: issue.id.clone(),
        content_hash: String::new(),
        title: String::new(),
        description: String::new(),
        design: String::new(),
        acceptance_criteria: String::new(),
        notes: String::new(),
        status: String::new(),
        priority: 0,
        issue_type: String::new(),
        assignee: None,
        estimated_minutes: None,
        created_at: issue.created_at.clone(),
        updated_at: issue.updated_at.clone(),
        closed_at: None,
        external_ref: None,
        sender: issue.sender.clone(),
        ephemeral: false,
        replies_to: String::new(),
        relates_to: Vec::new(),
        duplicate_of: String::new(),
        superseded_by: String::new(),
        deleted_at: None,
        deleted_by: String::new(),
        delete_reason: String::new(),
        original_type: String::new(),
        labels: Vec::new(),
        dependencies: Vec::new(),
        comments: Vec::new(),
        affected_symbols: Vec::new(),
        solid_volume: None,
        topology_hash: String::new(),
        is_solid: false,
    }
}

fn parse_timestamp(value: &str) -> Option<jiff::Timestamp> {
    value.parse().ok()
}

fn is_tombstone(issue: &Issue) -> bool {
    issue.status.trim().eq_ignore_ascii_case(STATUS_TOMBSTONE)
}

fn is_expired_tombstone(issue: &Issue, ttl_days: i64) -> bool {
    if !is_tombstone(issue) {
        return false;
    }
    let deleted_at = match issue.deleted_at.as_deref() {
        Some(value) => match parse_timestamp(value) {
            Some(parsed) => parsed,
            None => return false,
        },
        None => return false,
    };

    let ttl = if ttl_days == 0 {
        DEFAULT_TOMBSTONE_TTL_DAYS
    } else {
        ttl_days
    };
    let effective_ttl = ttl.days().hours(CLOCK_SKEW_GRACE_HOURS);
    let expiration = deleted_at + effective_ttl;
    jiff::Timestamp::now() > expiration
}

fn merge_tombstones(left: Issue, right: Issue) -> Issue {
    let left_deleted = left.deleted_at.as_deref().and_then(parse_timestamp);
    let right_deleted = right.deleted_at.as_deref().and_then(parse_timestamp);
    if left_deleted.is_none() && right_deleted.is_none() {
        return left;
    }
    if left_deleted.is_none() {
        return right;
    }
    if right_deleted.is_none() {
        return left;
    }
    if is_time_after(left_deleted, right_deleted) {
        left
    } else {
        right
    }
}

fn merge_issue(base: Issue, left: Issue, right: Issue) -> Issue {
    let mut result = base.clone();

    result.title = merge_field_by_updated_at(
        &base.title,
        &left.title,
        &right.title,
        &left.updated_at,
        &right.updated_at,
    );
    result.description = merge_field_by_updated_at(
        &base.description,
        &left.description,
        &right.description,
        &left.updated_at,
        &right.updated_at,
    );
    result.notes = merge_notes(&base.notes, &left.notes, &right.notes);
    result.status = merge_status(&base.status, &left.status, &right.status);
    result.priority = merge_priority(base.priority, left.priority, right.priority);
    result.issue_type = merge_field(&base.issue_type, &left.issue_type, &right.issue_type);

    result.updated_at = max_time_str(&left.updated_at, &right.updated_at);

    if result.status.trim().eq_ignore_ascii_case(STATUS_CLOSED) {
        result.closed_at = max_time_opt(left.closed_at.as_ref(), right.closed_at.as_ref());
    } else {
        result.closed_at = None;
    }

    result.dependencies = merge_dependencies(&left.dependencies, &right.dependencies);
    result.labels = merge_labels(&left.labels, &right.labels);
    result.comments = merge_comments(&left.comments, &right.comments);

    if result.status.trim().eq_ignore_ascii_case(STATUS_TOMBSTONE) {
        let left_deleted = left.deleted_at.as_deref().and_then(parse_timestamp);
        let right_deleted = right.deleted_at.as_deref().and_then(parse_timestamp);
        if is_time_after(left_deleted, right_deleted) {
            result.deleted_at = left.deleted_at;
            result.deleted_by = left.deleted_by;
            result.delete_reason = left.delete_reason;
            result.original_type = left.original_type;
        } else if right.deleted_at.is_some() {
            result.deleted_at = right.deleted_at;
            result.deleted_by = right.deleted_by;
            result.delete_reason = right.delete_reason;
            result.original_type = right.original_type;
        } else if left.deleted_at.is_some() {
            result.deleted_at = left.deleted_at;
            result.deleted_by = left.deleted_by;
            result.delete_reason = left.delete_reason;
            result.original_type = left.original_type;
        }
    }

    result
}

fn merge_field(base: &str, left: &str, right: &str) -> String {
    if base == left && base != right {
        return right.to_string();
    }
    if base == right && base != left {
        return left.to_string();
    }
    left.to_string()
}

fn merge_field_by_updated_at(
    base: &str,
    left: &str,
    right: &str,
    left_updated: &str,
    right_updated: &str,
) -> String {
    if base == left && base != right {
        return right.to_string();
    }
    if base == right && base != left {
        return left.to_string();
    }
    if left == right {
        return left.to_string();
    }
    let left_ts = parse_timestamp(left_updated);
    let right_ts = parse_timestamp(right_updated);
    match (left_ts, right_ts) {
        (Some(left_ts), Some(right_ts)) => {
            if left_ts > right_ts {
                left.to_string()
            } else {
                right.to_string()
            }
        }
        (Some(_), None) => left.to_string(),
        (None, Some(_)) => right.to_string(),
        (None, None) => left.to_string(),
    }
}

fn merge_notes(base: &str, left: &str, right: &str) -> String {
    if base == left && base != right {
        return right.to_string();
    }
    if base == right && base != left {
        return left.to_string();
    }
    if left == right {
        return left.to_string();
    }
    if left.is_empty() {
        return right.to_string();
    }
    if right.is_empty() {
        return left.to_string();
    }
    format!("{}\n\n---\n\n{}", left, right)
}

fn merge_status(base: &str, left: &str, right: &str) -> String {
    if left.trim().eq_ignore_ascii_case(STATUS_TOMBSTONE)
        || right.trim().eq_ignore_ascii_case(STATUS_TOMBSTONE)
    {
        return STATUS_TOMBSTONE.to_string();
    }
    if left.trim().eq_ignore_ascii_case(STATUS_CLOSED)
        || right.trim().eq_ignore_ascii_case(STATUS_CLOSED)
    {
        return STATUS_CLOSED.to_string();
    }
    merge_field(base, left, right)
}

fn merge_priority(base: i32, left: i32, right: i32) -> i32 {
    if base == left && base != right {
        return right;
    }
    if base == right && base != left {
        return left;
    }
    if left == right {
        return left;
    }
    if left == 0 && right != 0 {
        return right;
    }
    if right == 0 && left != 0 {
        return left;
    }
    if left < right { left } else { right }
}

fn merge_dependencies(left: &[Dependency], right: &[Dependency]) -> Vec<Dependency> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for dep in left.iter().chain(right.iter()) {
        let key = format!("{}:{}:{}", dep.issue_id, dep.depends_on_id, dep.type_);
        if seen.insert(key) {
            out.push(dep.clone());
        }
    }
    out
}

fn merge_labels(left: &[String], right: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for label in left.iter().chain(right.iter()) {
        if seen.insert(label.clone()) {
            out.push(label.clone());
        }
    }
    out
}

fn merge_comments(left: &[Comment], right: &[Comment]) -> Vec<Comment> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for comment in left.iter().chain(right.iter()) {
        let key = format!("{}:{}", comment.author, comment.text);
        if seen.insert(key) {
            out.push(comment.clone());
        }
    }
    out
}

fn is_time_after(left: Option<jiff::Timestamp>, right: Option<jiff::Timestamp>) -> bool {
    match (left, right) {
        (Some(left_ts), Some(right_ts)) => left_ts > right_ts,
        (Some(_), None) => true,
        (None, Some(_)) => false,
        (None, None) => false,
    }
}

fn max_time_str(left: &str, right: &str) -> String {
    match (parse_timestamp(left), parse_timestamp(right)) {
        (Some(left_ts), Some(right_ts)) => {
            if left_ts >= right_ts {
                left.to_string()
            } else {
                right.to_string()
            }
        }
        (Some(_), None) => left.to_string(),
        (None, Some(_)) => right.to_string(),
        (None, None) => {
            if left >= right {
                left.to_string()
            } else {
                right.to_string()
            }
        }
    }
}

fn max_time_opt(left: Option<&String>, right: Option<&String>) -> Option<String> {
    let left_ts = left.and_then(|value| parse_timestamp(value));
    let right_ts = right.and_then(|value| parse_timestamp(value));
    match (left, right, left_ts, right_ts) {
        (Some(left_raw), Some(right_raw), Some(left_parsed), Some(right_parsed)) => {
            if left_parsed >= right_parsed {
                Some(left_raw.clone())
            } else {
                Some(right_raw.clone())
            }
        }
        (Some(left_raw), Some(_), Some(_), None) => Some(left_raw.clone()),
        (Some(_), Some(right_raw), None, Some(_)) => Some(right_raw.clone()),
        (Some(left_raw), Some(right_raw), None, None) => {
            if left_raw >= right_raw {
                Some(left_raw.clone())
            } else {
                Some(right_raw.clone())
            }
        }
        (Some(left_raw), None, _, _) => Some(left_raw.clone()),
        (None, Some(right_raw), _, _) => Some(right_raw.clone()),
        (None, None, _, _) => None,
    }
}

pub async fn sync_with_git(
    db: &Db,
    repo_root: &Path,
    jsonl_path: &Path,
    message: &str,
    push: bool,
) -> Result<()> {
    export_to_jsonl(db, jsonl_path).await?;
    let rel_path = jsonl_path
        .strip_prefix(repo_root)
        .map_err(|_| eyre::eyre!("issues.jsonl must live under the repo root for git sync"))?;
    let rel_path = rel_path.to_string_lossy().replace('\\', "/");

    let upstream = git_output(
        repo_root,
        &["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
    );
    let upstream = match upstream {
        Ok(value) => value.trim().to_string(),
        Err(_) => {
            let changed = git_has_changes(repo_root, &rel_path)?;
            if changed {
                git_status(repo_root, &["add", &rel_path])?;
                git_status(repo_root, &["commit", "-m", message])?;
            }
            return Ok(());
        }
    };

    git_status(repo_root, &["fetch", "--prune"])?;

    let base = git_output(repo_root, &["merge-base", "HEAD", &upstream])?;
    let base = base.trim();
    let base_issues = read_issues_from_git(repo_root, base, &rel_path)?;
    let remote_issues = read_issues_from_git(repo_root, &upstream, &rel_path)?;
    let local_issues = read_issues_jsonl(jsonl_path)?;

    let merged = merge_issues_3way(&base_issues, &local_issues, &remote_issues);
    write_issues_jsonl(&merged, jsonl_path)?;
    sync_from_jsonl(db, jsonl_path).await?;

    let changed = git_has_changes(repo_root, &rel_path)?;
    if changed {
        git_status(repo_root, &["add", &rel_path])?;
        git_status(repo_root, &["commit", "-m", message])?;
    }

    if push {
        git_status(repo_root, &["push"])?;
    }

    Ok(())
}

fn git_output(repo_root: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo_root)
        .output()?;
    if !output.status.success() {
        return Err(eyre::eyre!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn git_status(repo_root: &Path, args: &[&str]) -> Result<()> {
    let status = Command::new("git")
        .args(args)
        .current_dir(repo_root)
        .status()?;
    if !status.success() {
        return Err(eyre::eyre!("git {} failed", args.join(" ")));
    }
    Ok(())
}

fn git_has_changes(repo_root: &Path, rel_path: &str) -> Result<bool> {
    let output = git_output(repo_root, &["status", "--porcelain", "--", rel_path])?;
    Ok(!output.trim().is_empty())
}

fn read_issues_from_git(repo_root: &Path, reference: &str, rel_path: &str) -> Result<Vec<Issue>> {
    let spec = format!("{reference}:{rel_path}");
    let exists = Command::new("git")
        .args(["cat-file", "-e", &spec])
        .current_dir(repo_root)
        .status()?
        .success();
    if !exists {
        return Ok(Vec::new());
    }
    let content = git_output(repo_root, &["show", &spec])?;
    let mut issues = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let issue: Issue = serde_json::from_str(trimmed)?;
        issues.push(issue);
    }
    Ok(issues)
}

pub fn normalize_tags(values: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for value in values {
        let normalized = value.trim().to_ascii_lowercase();
        if normalized.is_empty() || !seen.insert(normalized.clone()) {
            continue;
        }
        out.push(normalized);
    }
    out.sort();
    out
}

pub fn build_dependencies(issue_id: &str, values: &[String], created_by: &str) -> Vec<Dependency> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for value in values {
        let (depends_on_id, type_) = parse_dependency_spec(value);
        let normalized = depends_on_id.trim().to_ascii_lowercase();
        if normalized.is_empty() || !seen.insert(normalized.clone()) {
            continue;
        }
        out.push(Dependency {
            issue_id: issue_id.to_string(),
            depends_on_id,
            type_,
            created_at: now_timestamp(),
            created_by: created_by.to_string(),
        });
    }
    out
}

pub fn apply_dependency_changes(
    existing: &[Dependency],
    add: &[String],
    remove: &[String],
    issue_id: &str,
    created_by: &str,
) -> Vec<Dependency> {
    let mut out: Vec<Dependency> = existing.to_vec();
    for value in add {
        let (depends_on_id, type_) = parse_dependency_spec(value);
        if depends_on_id.trim().is_empty() {
            continue;
        }
        if out
            .iter()
            .any(|dep| dep.depends_on_id == depends_on_id && dep.type_ == type_)
        {
            continue;
        }
        out.push(Dependency {
            issue_id: issue_id.to_string(),
            depends_on_id,
            type_,
            created_at: now_timestamp(),
            created_by: created_by.to_string(),
        });
    }
    for value in remove {
        let (depends_on_id, _) = parse_dependency_spec(value);
        if depends_on_id.trim().is_empty() {
            continue;
        }
        out.retain(|dep| dep.depends_on_id != depends_on_id);
    }
    out
}

pub fn normalize_ids(values: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for value in values {
        let normalized = value.trim().to_ascii_lowercase();
        if normalized.is_empty() || !seen.insert(normalized.clone()) {
            continue;
        }
        out.push(normalized);
    }
    out.sort();
    out
}

pub fn normalize_symbols(values: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for value in values {
        let normalized = value
            .trim()
            .replace('\\', "/")
            .trim_end_matches('/')
            .to_string();
        if normalized.is_empty() || !seen.insert(normalized.clone()) {
            continue;
        }
        out.push(normalized);
    }
    out.sort();
    out
}

pub fn merge_tags(existing: &[String], add: &[String], remove: &[String]) -> Vec<String> {
    let mut set: HashSet<String> = existing.iter().map(|v| v.to_ascii_lowercase()).collect();
    for value in normalize_tags(add) {
        set.insert(value);
    }
    for value in normalize_tags(remove) {
        set.remove(&value);
    }
    let mut out: Vec<String> = set.into_iter().collect();
    out.sort();
    out
}

pub fn merge_ids(existing: &[String], add: &[String], remove: &[String]) -> Vec<String> {
    let mut set: HashSet<String> = existing.iter().map(|v| v.to_ascii_lowercase()).collect();
    for value in normalize_ids(add) {
        set.insert(value);
    }
    for value in normalize_ids(remove) {
        set.remove(&value);
    }
    let mut out: Vec<String> = set.into_iter().collect();
    out.sort();
    out
}

pub fn merge_symbols(existing: &[String], add: &[String], remove: &[String]) -> Vec<String> {
    let mut set: HashSet<String> = existing.iter().cloned().collect();
    for value in normalize_symbols(add) {
        set.insert(value);
    }
    for value in normalize_symbols(remove) {
        set.remove(&value);
    }
    let mut out: Vec<String> = set.into_iter().collect();
    out.sort();
    out
}

fn now_timestamp() -> String {
    jiff::Timestamp::now().to_string()
}

pub fn build_comments(issue_id: &str, values: &[String], author: &str) -> Vec<Comment> {
    let mut out = Vec::new();
    for value in values {
        let text = value.trim().to_string();
        if text.is_empty() {
            continue;
        }
        let now = now_timestamp();
        out.push(Comment {
            id: Uuid::now_v7().to_string(),
            issue_id: issue_id.to_string(),
            author: author.to_string(),
            text,
            created_at: now,
        });
    }
    out
}

fn parse_dependency_spec(value: &str) -> (String, String) {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return (String::new(), "blocking".to_string());
    }
    let mut parts = trimmed.splitn(2, ':');
    let id = parts.next().unwrap_or("").trim().to_string();
    let kind = parts
        .next()
        .unwrap_or("blocking")
        .trim()
        .to_ascii_lowercase();
    let kind = if kind.is_empty() {
        "blocking".to_string()
    } else {
        kind
    };
    (id, kind)
}
