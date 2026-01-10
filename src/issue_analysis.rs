use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use eyre::{Result, eyre};
use jiff::{Timestamp, Unit};

use crate::issues::Issue;

#[derive(Clone, Debug)]
pub struct NextTaskSuggestion {
    pub issue: Issue,
    pub reason: String,
    pub blockers: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct StaleIssue {
    pub issue: Issue,
    pub days_inactive: i64,
}

#[derive(Clone, Debug)]
pub struct DuplicatePair {
    pub issue_a: Issue,
    pub issue_b: Issue,
    pub similarity: f64,
}

#[derive(Clone, Debug)]
pub struct RelatedIssue {
    pub issue: Issue,
    pub score: f64,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct IssueDraft {
    pub title: String,
    pub issue_type: String,
    pub description: String,
    pub affected_symbols: Vec<String>,
    pub labels: Vec<String>,
}

pub fn suggest_next_tasks(
    issues: &[Issue],
    assignee: Option<&str>,
    limit: usize,
) -> Result<Vec<NextTaskSuggestion>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let mut status_map = HashMap::new();
    for issue in issues {
        status_map.insert(issue.id.clone(), issue.status.clone());
    }

    let mut candidates = Vec::new();
    for issue in issues {
        if issue.status == "closed" || issue.status == "blocked" {
            continue;
        }
        if let Some(assignee_filter) = assignee {
            if issue.assignee.as_deref() != Some(assignee_filter) {
                continue;
            }
        }

        let mut blockers = Vec::new();
        for dep in &issue.dependencies {
            if dep.id.trim().is_empty() || !dep.is_blocking() {
                continue;
            }
            match status_map.get(&dep.id) {
                Some(status) if status != "closed" => blockers.push(dep.id.clone()),
                None => blockers.push(dep.id.clone()),
                _ => {}
            }
        }
        if !blockers.is_empty() {
            continue;
        }

        let age_days = days_since(&issue.updated_at)?;
        let mut reason = format!("priority={} updated={}d", issue.priority, age_days);
        if issue.assignee.is_none() {
            reason.push_str(" unassigned");
        }
        candidates.push(NextTaskSuggestion {
            issue: issue.clone(),
            reason,
            blockers,
        });
    }

    candidates.sort_by(|a, b| {
        a.issue
            .priority
            .cmp(&b.issue.priority)
            .then_with(|| {
                let a_days = days_since(&a.issue.updated_at).unwrap_or(0);
                let b_days = days_since(&b.issue.updated_at).unwrap_or(0);
                b_days.cmp(&a_days)
            })
            .then_with(|| a.issue.id.cmp(&b.issue.id))
    });

    if candidates.len() > limit {
        candidates.truncate(limit);
    }
    Ok(candidates)
}

pub fn find_stale_issues(
    issues: &[Issue],
    days: i64,
    assignee: Option<&str>,
    status: Option<&str>,
    limit: usize,
) -> Result<Vec<StaleIssue>> {
    let mut results = Vec::new();
    for issue in issues {
        if issue.status == "closed" {
            continue;
        }
        if let Some(status_filter) = status {
            if issue.status != status_filter {
                continue;
            }
        }
        if let Some(assignee_filter) = assignee {
            if issue.assignee.as_deref() != Some(assignee_filter) {
                continue;
            }
        }
        let age_days = days_since(&issue.updated_at)?;
        if age_days >= days {
            results.push(StaleIssue {
                issue: issue.clone(),
                days_inactive: age_days,
            });
        }
    }

    results.sort_by(|a, b| {
        b.days_inactive
            .cmp(&a.days_inactive)
            .then_with(|| a.issue.id.cmp(&b.issue.id))
    });
    if limit > 0 && results.len() > limit {
        results.truncate(limit);
    }
    Ok(results)
}

pub fn find_duplicates(issues: &[Issue], threshold: f64, limit: usize) -> Vec<DuplicatePair> {
    if issues.len() < 2 || limit == 0 {
        return Vec::new();
    }

    let mut tokens = Vec::with_capacity(issues.len());
    for issue in issues {
        if issue.status == "closed" {
            tokens.push(None);
            continue;
        }
        let text = issue.summary_query();
        let token_set = tokenize(&text);
        tokens.push(Some(token_set));
    }

    let mut pairs = Vec::new();
    for i in 0..issues.len() {
        let Some(tokens_a) = tokens[i].as_ref() else {
            continue;
        };
        for j in (i + 1)..issues.len() {
            let Some(tokens_b) = tokens[j].as_ref() else {
                continue;
            };
            let similarity = jaccard(tokens_a, tokens_b);
            if similarity >= threshold {
                pairs.push(DuplicatePair {
                    issue_a: issues[i].clone(),
                    issue_b: issues[j].clone(),
                    similarity,
                });
            }
        }
    }

    pairs.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if pairs.len() > limit {
        pairs.truncate(limit);
    }
    pairs
}

pub fn related_issues_for_file(
    issues: &[Issue],
    file_path: &str,
    limit: usize,
) -> Vec<RelatedIssue> {
    let mut results = Vec::new();
    let target = normalize_path(file_path);
    let target_lower = target.to_ascii_lowercase();
    for issue in issues {
        if issue.status == "closed" {
            continue;
        }
        let mut reasons = Vec::new();
        let mut score = 0.0;

        for symbol in &issue.affected_symbols {
            let normalized = normalize_path(symbol);
            if normalized == target || normalized.starts_with(&format!("{target}/")) {
                reasons.push("affected_symbols".to_string());
                score += 1.0;
                break;
            }
        }

        let summary = issue.summary_query().to_ascii_lowercase();
        if summary.contains(&target_lower) {
            reasons.push("text_match".to_string());
            score += 0.5;
        }

        if score > 0.0 {
            results.push(RelatedIssue {
                issue: issue.clone(),
                score,
                reasons,
            });
        }
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.issue.id.cmp(&b.issue.id))
    });
    if limit > 0 && results.len() > limit {
        results.truncate(limit);
    }
    results
}

pub fn related_issues_for_issue(
    issues: &[Issue],
    target: &Issue,
    limit: usize,
) -> Vec<RelatedIssue> {
    let mut results = Vec::new();
    let target_symbols: HashSet<String> = target
        .affected_symbols
        .iter()
        .map(|s| normalize_path(s))
        .collect();
    let target_labels: HashSet<String> = target
        .labels
        .iter()
        .map(|label| label.to_ascii_lowercase())
        .collect();
    let target_dependencies: HashSet<String> =
        target.dependency_ids().into_iter().collect();
    let target_tokens = tokenize(&target.summary_query());

    for issue in issues {
        if issue.id == target.id || issue.status == "closed" {
            continue;
        }
        let mut reasons = Vec::new();
        let mut score = 0.0;

        let shared_symbols = issue
            .affected_symbols
            .iter()
            .map(|s| normalize_path(s))
            .filter(|symbol| target_symbols.contains(symbol))
            .count();
        if shared_symbols > 0 {
            reasons.push(format!("shared_symbols={shared_symbols}"));
            score += 1.0 + (shared_symbols as f64 * 0.1);
        }

        let shared_labels = issue
            .labels
            .iter()
            .map(|label| label.to_ascii_lowercase())
            .filter(|label| target_labels.contains(label))
            .count();
        if shared_labels > 0 {
            reasons.push(format!("shared_labels={shared_labels}"));
            score += 0.3 + (shared_labels as f64 * 0.05);
        }

        if target_dependencies.contains(&issue.id) {
            reasons.push("dependency".to_string());
            score += 1.0;
        }
        if target.relates_to.iter().any(|id| id == &issue.id) {
            reasons.push("relates_to".to_string());
            score += 0.7;
        }

        let similarity = jaccard(&target_tokens, &tokenize(&issue.summary_query()));
        if similarity > 0.0 {
            reasons.push(format!("text_similarity={similarity:.2}"));
            score += similarity * 0.5;
        }

        if score > 0.0 {
            results.push(RelatedIssue {
                issue: issue.clone(),
                score,
                reasons,
            });
        }
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.issue.id.cmp(&b.issue.id))
    });
    if limit > 0 && results.len() > limit {
        results.truncate(limit);
    }
    results
}

pub fn infer_issue_from_error(message: &str) -> IssueDraft {
    let summary = first_non_empty_line(message).unwrap_or_else(|| "Runtime error".to_string());
    let affected_symbols = extract_paths(message);
    IssueDraft {
        title: summary,
        issue_type: "bug".to_string(),
        description: message.trim().to_string(),
        affected_symbols,
        labels: vec!["error".to_string()],
    }
}

pub fn infer_issue_from_diff(diff: &str) -> IssueDraft {
    let files = extract_paths(diff);
    let title = if files.is_empty() {
        "Investigate recent code changes".to_string()
    } else if files.len() == 1 {
        format!("Update {}", files[0])
    } else {
        format!("Update {} files", files.len())
    };
    let description = if files.is_empty() {
        "Changes detected in the working tree.".to_string()
    } else {
        format!("Changed files:\n{}", files.join("\n"))
    };
    IssueDraft {
        title,
        issue_type: "task".to_string(),
        description,
        affected_symbols: files,
        labels: Vec::new(),
    }
}

pub fn infer_issues_from_todos(file_path: &str, content: &str, limit: usize) -> Vec<IssueDraft> {
    let mut drafts = Vec::new();
    for (idx, line) in content.lines().enumerate() {
        if !line.to_ascii_lowercase().contains("todo") {
            continue;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let title = trimmed
            .trim_start_matches(|ch: char| ch == '/' || ch == '*' || ch == '#')
            .trim()
            .trim_start_matches("TODO")
            .trim_start_matches(':')
            .trim()
            .to_string();
        let title = if title.is_empty() {
            format!("TODO in {}", file_path)
        } else {
            title
        };
        drafts.push(IssueDraft {
            title,
            issue_type: "task".to_string(),
            description: format!("TODO in {} at line {}:\n{}", file_path, idx + 1, trimmed),
            affected_symbols: vec![normalize_path(file_path)],
            labels: vec!["todo".to_string()],
        });
        if limit > 0 && drafts.len() >= limit {
            break;
        }
    }
    drafts
}

pub fn extract_paths(text: &str) -> Vec<String> {
    let mut paths = BTreeSet::new();
    for token in text.split_whitespace() {
        let candidate = token
            .trim_matches(|ch: char| {
                ch == '"'
                    || ch == '\''
                    || ch == ','
                    || ch == ':'
                    || ch == ';'
                    || ch == ')'
                    || ch == '('
            })
            .trim();
        if candidate.is_empty() {
            continue;
        }
        let candidate = candidate.trim_end_matches(|ch: char| ch == '.' || ch == ']' || ch == '}');
        let candidate = strip_line_suffix(candidate);
        if candidate.contains('/') || candidate.contains('\\') {
            if candidate.contains('.') || candidate.contains("::") {
                paths.insert(normalize_path(&candidate));
            }
        }
    }
    paths.into_iter().collect()
}

fn days_since(timestamp: &str) -> Result<i64> {
    let parsed: Timestamp = timestamp
        .parse()
        .map_err(|err| eyre!("invalid timestamp '{}': {}", timestamp, err))?;
    let now = Timestamp::now();
    let span = now.since(parsed)?;
    let seconds = span.total(Unit::Second)?;
    Ok((seconds.abs() / 86_400.0).floor() as i64)
}

fn tokenize(text: &str) -> HashSet<String> {
    let mut tokens = HashSet::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            tokens.insert(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.insert(current);
    }
    tokens
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn normalize_path(path: &str) -> String {
    path.trim()
        .replace('\\', "/")
        .trim_start_matches("./")
        .trim_end_matches('/')
        .to_string()
}

fn first_non_empty_line(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(str::to_string)
}

fn strip_line_suffix(value: &str) -> String {
    let mut base = value;
    let mut rest = value;
    let mut removed = 0;
    loop {
        let Some((left, right)) = rest.rsplit_once(':') else {
            break;
        };
        if right.chars().all(|ch| ch.is_ascii_digit()) {
            base = left;
            rest = left;
            removed += 1;
            if removed >= 2 {
                break;
            }
            continue;
        }
        break;
    }
    base.to_string()
}

pub fn group_related_issues_by_file(
    issues: &[Issue],
    files: &[String],
    limit: usize,
) -> BTreeMap<String, Vec<RelatedIssue>> {
    let mut map = BTreeMap::new();
    for file in files {
        let related = related_issues_for_file(issues, file, limit);
        if !related.is_empty() {
            map.insert(file.clone(), related);
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_issue(id: &str, title: &str) -> Issue {
        let mut issue = Issue::new(title.to_string());
        issue.id = id.to_string();
        issue
    }

    #[test]
    fn suggest_next_tasks_skips_blocked_and_closed() -> Result<()> {
        let mut ready = make_issue("cr-ready", "Ready");
        ready.priority = 1;
        ready.updated_at = Timestamp::now().to_string();

        let mut blocked = make_issue("cr-blocked", "Blocked");
        blocked.dependencies = vec![crate::issues::IssueDependency {
            id: "cr-missing".to_string(),
            kind: "blocks".to_string(),
        }];

        let mut closed = make_issue("cr-closed", "Closed");
        closed.status = "closed".to_string();

        let suggestions = suggest_next_tasks(&[ready.clone(), blocked, closed], None, 10)?;
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].issue.id, ready.id);
        assert!(suggestions[0].reason.contains("priority=1"));
        Ok(())
    }

    #[test]
    fn find_stale_issues_respects_threshold() -> Result<()> {
        let mut old = make_issue("cr-old", "Old");
        old.updated_at = "2000-01-01T00:00:00Z".to_string();

        let mut fresh = make_issue("cr-fresh", "Fresh");
        fresh.updated_at = Timestamp::now().to_string();

        let stale = find_stale_issues(&[old.clone(), fresh], 7, None, None, 10)?;
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].issue.id, old.id);
        Ok(())
    }

    #[test]
    fn find_duplicates_ignores_closed() {
        let mut first = make_issue("cr-a", "Retry API calls");
        first.description = "Add retry logic for timeouts".to_string();

        let mut second = make_issue("cr-b", "Add retry logic");
        second.description = "Retry API calls on timeout".to_string();

        let mut closed = make_issue("cr-c", "Add retry logic");
        closed.status = "closed".to_string();

        let pairs = find_duplicates(&[first.clone(), second.clone(), closed], 0.2, 10);
        assert_eq!(pairs.len(), 1);
        let ids = vec![pairs[0].issue_a.id.clone(), pairs[0].issue_b.id.clone()];
        assert!(ids.contains(&first.id));
        assert!(ids.contains(&second.id));
    }

    #[test]
    fn related_issues_for_file_prefers_symbol_matches() {
        let mut issue = make_issue("cr-symbol", "Fix topology");
        issue.affected_symbols = vec!["src/topology.rs".to_string()];

        let mut text_only = make_issue("cr-text", "Investigate src/topology.rs errors");
        text_only.description = "path mentioned in text".to_string();

        let related = related_issues_for_file(&[issue.clone(), text_only], "src/topology.rs", 10);
        assert_eq!(related.len(), 2);
        assert_eq!(related[0].issue.id, issue.id);
    }

    #[test]
    fn related_issues_for_issue_scores_shared_metadata() {
        let mut target = make_issue("cr-target", "Search ranking bug");
        target.labels = vec!["search".to_string()];
        target.affected_symbols = vec!["src/search.rs".to_string()];

        let mut candidate = make_issue("cr-related", "Search results wrong");
        candidate.labels = vec!["search".to_string()];
        candidate.affected_symbols = vec!["src/search.rs".to_string()];

        let related = related_issues_for_issue(&[target.clone(), candidate.clone()], &target, 10);
        assert_eq!(related.len(), 1);
        assert_eq!(related[0].issue.id, candidate.id);
        assert!(
            related[0]
                .reasons
                .iter()
                .any(|reason| reason.starts_with("shared_symbols"))
        );
    }

    #[test]
    fn infer_issue_from_error_extracts_path() {
        let message = "panic: failed to parse src/main.rs:12";
        let draft = infer_issue_from_error(message);
        assert_eq!(draft.issue_type, "bug");
        assert!(draft.affected_symbols.contains(&"src/main.rs".to_string()));
        assert_eq!(draft.title, "panic: failed to parse src/main.rs:12");
    }

    #[test]
    fn infer_issue_from_diff_uses_file_list() {
        let diff = "+++ b/src/lib.rs\n";
        let draft = infer_issue_from_diff(diff);
        assert_eq!(draft.title, "Update b/src/lib.rs");
        assert_eq!(draft.affected_symbols, vec!["b/src/lib.rs".to_string()]);
    }

    #[test]
    fn infer_issues_from_todos_builds_drafts() {
        let content = "// TODO: cleanup error handling\nfn main() {}\n";
        let drafts = infer_issues_from_todos("src/lib.rs", content, 10);
        assert_eq!(drafts.len(), 1);
        assert!(drafts[0].title.contains("cleanup"));
        assert_eq!(drafts[0].labels, vec!["todo".to_string()]);
    }
}
