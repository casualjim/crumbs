use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use eyre::Result;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::db::Db;

const ISSUE_PREFIX: &str = "cr";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IssueDependency {
    pub id: String,
    #[serde(default)]
    pub kind: String,
}

impl IssueDependency {
    pub fn is_blocking(&self) -> bool {
        let kind = self.kind.trim().to_ascii_lowercase();
        !matches!(kind.as_str(), "related" | "optional" | "informational")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IssueComment {
    pub id: String,
    pub body: String,
    pub author: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IssueExternalRef {
    pub kind: String,
    pub value: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Issue {
    pub id: String,
    pub title: String,
    pub description: String,
    pub design: String,
    pub acceptance_criteria: String,
    pub notes: String,
    pub status: String,
    pub priority: i32,
    pub issue_type: String,
    pub assignee: Option<String>,
    pub labels: Vec<String>,
    #[serde(default, deserialize_with = "deserialize_dependencies")]
    pub dependencies: Vec<IssueDependency>,
    pub relates_to: Vec<String>,
    pub affected_symbols: Vec<String>,
    #[serde(default)]
    pub external_refs: Vec<IssueExternalRef>,
    #[serde(default)]
    pub comments: Vec<IssueComment>,
    pub estimate_minutes: Option<i32>,
    pub duplicate_of: Option<String>,
    pub superseded_by: Option<String>,
    pub deleted_at: Option<String>,
    pub deleted_by: Option<String>,
    pub deleted_reason: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub closed_at: Option<String>,
}

impl Issue {
    pub fn new(title: String) -> Self {
        let now = now_timestamp();
        Self {
            id: generate_issue_id(),
            title,
            description: String::new(),
            design: String::new(),
            acceptance_criteria: String::new(),
            notes: String::new(),
            status: "open".to_string(),
            priority: 3,
            issue_type: "task".to_string(),
            assignee: None,
            labels: Vec::new(),
            dependencies: Vec::new(),
            relates_to: Vec::new(),
            affected_symbols: Vec::new(),
            external_refs: Vec::new(),
            comments: Vec::new(),
            estimate_minutes: None,
            duplicate_of: None,
            superseded_by: None,
            deleted_at: None,
            deleted_by: None,
            deleted_reason: None,
            created_at: now.clone(),
            updated_at: now,
            closed_at: None,
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
            .map(|dep| dep.id.clone())
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
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let issue: Issue = serde_json::from_str(trimmed)?;
        db.upsert_issue(&issue).await?;
    }
    Ok(())
}

pub async fn export_to_jsonl(db: &Db, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let issues = db.list_all_issues().await?;
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for issue in issues {
        let line = serde_json::to_string(&issue)?;
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
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

pub fn normalize_dependencies(values: &[String]) -> Vec<IssueDependency> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for value in values {
        let (id, kind) = parse_dependency_spec(value);
        let normalized = id.trim().to_ascii_lowercase();
        if normalized.is_empty() || !seen.insert(normalized) {
            continue;
        }
        out.push(IssueDependency { id, kind });
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    out
}

pub fn merge_dependencies(
    existing: &[IssueDependency],
    add: &[String],
    remove: &[String],
) -> Vec<IssueDependency> {
    let mut map: HashMap<String, IssueDependency> = existing
        .iter()
        .map(|dep| (dep.id.to_ascii_lowercase(), dep.clone()))
        .collect();
    for value in add {
        let (id, kind) = parse_dependency_spec(value);
        let normalized = id.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            continue;
        }
        map.insert(normalized, IssueDependency { id, kind });
    }
    for value in remove {
        let (id, _) = parse_dependency_spec(value);
        let normalized = id.trim().to_ascii_lowercase();
        map.remove(&normalized);
    }
    let mut out: Vec<IssueDependency> = map.into_values().collect();
    out.sort_by(|a, b| a.id.cmp(&b.id));
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

pub fn normalize_external_refs(values: &[String]) -> Vec<IssueExternalRef> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for value in values {
        let (kind, reference) = parse_external_ref_spec(value);
        let normalized = reference.trim().to_ascii_lowercase();
        if normalized.is_empty() || !seen.insert(normalized.clone()) {
            continue;
        }
        out.push(IssueExternalRef { kind, value: reference });
    }
    out.sort_by(|a, b| a.value.cmp(&b.value));
    out
}

pub fn merge_external_refs(
    existing: &[IssueExternalRef],
    add: &[String],
    remove: &[String],
) -> Vec<IssueExternalRef> {
    let mut map: HashMap<String, IssueExternalRef> = existing
        .iter()
        .map(|reference| (reference.value.to_ascii_lowercase(), reference.clone()))
        .collect();
    for value in add {
        let (kind, reference) = parse_external_ref_spec(value);
        let normalized = reference.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            continue;
        }
        map.insert(normalized, IssueExternalRef { kind, value: reference });
    }
    for value in remove {
        let (_kind, reference) = parse_external_ref_spec(value);
        let normalized = reference.trim().to_ascii_lowercase();
        map.remove(&normalized);
    }
    let mut out: Vec<IssueExternalRef> = map.into_values().collect();
    out.sort_by(|a, b| a.value.cmp(&b.value));
    out
}

pub fn merge_tags(
    existing: &[String],
    add: &[String],
    remove: &[String],
) -> Vec<String> {
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

pub fn merge_ids(
    existing: &[String],
    add: &[String],
    remove: &[String],
) -> Vec<String> {
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

pub fn merge_symbols(
    existing: &[String],
    add: &[String],
    remove: &[String],
) -> Vec<String> {
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

fn generate_issue_id() -> String {
    let raw = Uuid::now_v7().to_string().replace('-', "");
    let suffix = raw.get(0..8).unwrap_or(&raw);
    format!("{ISSUE_PREFIX}-{suffix}")
}

fn now_timestamp() -> String {
    jiff::Timestamp::now().to_string()
}

pub fn build_comments(values: &[String], author: Option<&str>) -> Vec<IssueComment> {
    let mut out = Vec::new();
    for value in values {
        let body = value.trim().to_string();
        if body.is_empty() {
            continue;
        }
        let now = now_timestamp();
        out.push(IssueComment {
            id: Uuid::now_v7().to_string(),
            body,
            author: author.map(|value| value.to_string()).filter(|value| !value.trim().is_empty()),
            created_at: now.clone(),
            updated_at: now,
        });
    }
    out
}

fn parse_dependency_spec(value: &str) -> (String, String) {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return (String::new(), "blocks".to_string());
    }
    let mut parts = trimmed.splitn(2, ':');
    let id = parts.next().unwrap_or("").trim().to_string();
    let kind = parts
        .next()
        .unwrap_or("blocks")
        .trim()
        .to_ascii_lowercase();
    let kind = if kind.is_empty() { "blocks".to_string() } else { kind };
    (id, kind)
}

fn parse_external_ref_spec(value: &str) -> (String, String) {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return ("external".to_string(), String::new());
    }
    if trimmed.contains("://") {
        return ("url".to_string(), trimmed.to_string());
    }
    let mut parts = trimmed.splitn(2, ':');
    let first = parts.next().unwrap_or("").trim().to_string();
    let second = parts.next().map(str::trim).unwrap_or("");
    if second.is_empty() {
        ("external".to_string(), first)
    } else {
        (first.to_ascii_lowercase(), second.to_string())
    }
}

fn deserialize_dependencies<'de, D>(deserializer: D) -> Result<Vec<IssueDependency>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum DependencyList {
        Strings(Vec<String>),
        Structured(Vec<IssueDependency>),
    }

    let list = DependencyList::deserialize(deserializer)?;
    let deps = match list {
        DependencyList::Strings(values) => normalize_dependencies(&values),
        DependencyList::Structured(values) => values,
    };
    Ok(deps)
}
