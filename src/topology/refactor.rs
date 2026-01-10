use std::fs;
use std::path::{Path, PathBuf};

use eyre::{Result, bail};

#[derive(Clone, Debug)]
pub enum RefactorKind {
    CommentOut,
    AddWarningComment,
}

#[derive(Clone, Debug)]
pub struct RefactorAction {
    pub file_path: String,
    pub line_start: usize,
    pub line_end: usize,
    pub action: RefactorKind,
    pub original_code: String,
    pub modified_code: String,
    pub edge_from: String,
    pub edge_to: String,
    pub reasoning: String,
}

impl RefactorAction {
    pub fn comment_out(
        file_path: &str,
        line_start: usize,
        line_end: usize,
        original_code: &str,
        edge_from: &str,
        edge_to: &str,
    ) -> Self {
        let comment_prefix = if file_path.ends_with(".py") {
            "# "
        } else {
            "// "
        };

        let modified_lines: Vec<String> = original_code
            .lines()
            .map(|line| format!("{comment_prefix}{line}"))
            .collect();

        let warning = format!(
            "{prefix}TODO(crumbs): Commented out to break cycle: {edge_from} -> {edge_to}",
            prefix = comment_prefix,
            edge_from = edge_from,
            edge_to = edge_to
        );

        let modified_code = format!("{}\n{}", warning, modified_lines.join("\n"));

        Self {
            file_path: file_path.to_string(),
            line_start,
            line_end,
            action: RefactorKind::CommentOut,
            original_code: original_code.to_string(),
            modified_code,
            edge_from: edge_from.to_string(),
            edge_to: edge_to.to_string(),
            reasoning: format!(
                "This edge ({} -> {}) was identified as the weakest link in the dependency cycle",
                edge_from, edge_to
            ),
        }
    }

    pub fn preview_diff(&self) -> String {
        let mut diff = String::new();
        diff.push_str(&format!("--- a/{}\n", self.file_path));
        diff.push_str(&format!("+++ b/{}\n", self.file_path));
        diff.push_str(&format!(
            "@@ -{},{} +{},{} @@\n",
            self.line_start,
            self.original_code.lines().count(),
            self.line_start,
            self.modified_code.lines().count()
        ));

        for line in self.original_code.lines() {
            diff.push_str(&format!("-{}\n", line));
        }
        for line in self.modified_code.lines() {
            diff.push_str(&format!("+{}\n", line));
        }

        diff
    }

    pub fn apply(&self, backup_dir: Option<&Path>) -> Result<()> {
        let file_path = Path::new(&self.file_path);
        let content = fs::read_to_string(file_path)?;
        let lines: Vec<&str> = content.lines().collect();

        if self.line_start < 1 || self.line_end > lines.len() || self.line_start > self.line_end {
            bail!(
                "invalid line range {}-{} for file with {} lines",
                self.line_start,
                self.line_end,
                lines.len()
            );
        }

        if let Some(dir) = backup_dir {
            fs::create_dir_all(dir)?;
            let backup_path = dir
                .join(file_path.file_name().unwrap_or_default())
                .with_extension("bak");
            fs::write(backup_path, &content)?;
        }

        let mut new_lines = Vec::new();
        for (idx, line) in lines.iter().enumerate() {
            let line_num = idx + 1;
            if line_num == self.line_start {
                new_lines.extend(self.modified_code.lines().map(|line| line.to_string()));
            }
            if line_num < self.line_start || line_num > self.line_end {
                new_lines.push((*line).to_string());
            }
        }

        let updated = new_lines.join("\n");
        fs::write(file_path, updated)?;
        Ok(())
    }
}

pub fn undo_refactor(file_path: &str, backup_dir: &Path) -> Result<()> {
    let file = Path::new(file_path);
    let backup_file = backup_dir
        .join(file.file_name().unwrap_or_default())
        .with_extension("bak");

    if !backup_file.exists() {
        bail!("no backup found for {}", file_path);
    }

    let backup_content = fs::read_to_string(&backup_file)?;
    fs::write(file, backup_content)?;
    fs::remove_file(&backup_file)?;
    Ok(())
}

pub fn get_backup_dir(project_root: &Path) -> PathBuf {
    project_root.join(".config").join("crumbs").join("backups")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_comment_out_rust() {
        let action = RefactorAction::comment_out(
            "src/main.rs",
            10,
            10,
            "use crate::foo::bar;",
            "main.rs::main",
            "foo.rs::bar",
        );

        assert!(action.modified_code.contains("// TODO(crumbs)"));
        assert!(action.modified_code.contains("// use crate::foo::bar;"));
        assert!(matches!(action.action, RefactorKind::CommentOut));
        assert!(action.reasoning.contains("weakest link"));
        assert_eq!(action.edge_from, "main.rs::main");
        assert_eq!(action.edge_to, "foo.rs::bar");
    }

    #[test]
    fn test_comment_out_python() {
        let action = RefactorAction::comment_out(
            "src/main.py",
            5,
            5,
            "from foo import bar",
            "main.py::main",
            "foo.py::bar",
        );

        assert!(action.modified_code.contains("# TODO(crumbs)"));
        assert!(action.modified_code.contains("# from foo import bar"));
        assert!(matches!(action.action, RefactorKind::CommentOut));
    }

    #[test]
    fn test_preview_diff() {
        let action = RefactorAction::comment_out("src/test.rs", 1, 1, "use foo;", "a", "b");

        let diff = action.preview_diff();
        assert!(diff.contains("--- a/src/test.rs"));
        assert!(diff.contains("+++ b/src/test.rs"));
        assert!(diff.contains("-use foo;"));
        assert!(diff.contains("+// TODO(crumbs)"));
    }

    #[test]
    fn test_apply_and_undo_refactor() -> Result<()> {
        let dir = TempDir::new()?;
        let root = dir.path();
        let file_path = root.join("main.rs");
        std::fs::write(&file_path, "use foo;\n")?;

        let action = RefactorAction::comment_out(
            file_path.to_string_lossy().as_ref(),
            1,
            1,
            "use foo;",
            "a",
            "b",
        );

        let backup_dir = get_backup_dir(root);
        action.apply(Some(&backup_dir))?;

        let updated = std::fs::read_to_string(&file_path)?;
        assert!(updated.contains("TODO(crumbs)"));

        undo_refactor(file_path.to_string_lossy().as_ref(), &backup_dir)?;
        let restored = std::fs::read_to_string(&file_path)?;
        assert_eq!(restored, "use foo;\n");

        Ok(())
    }

    #[test]
    fn test_add_warning_comment_variant() {
        let action = RefactorAction {
            file_path: "src/lib.rs".to_string(),
            line_start: 1,
            line_end: 1,
            action: RefactorKind::AddWarningComment,
            original_code: "fn main() {}".to_string(),
            modified_code: "fn main() {}".to_string(),
            edge_from: "a".to_string(),
            edge_to: "b".to_string(),
            reasoning: "test".to_string(),
        };

        assert!(matches!(action.action, RefactorKind::AddWarningComment));
        assert_eq!(action.line_end, 1);
    }
}
