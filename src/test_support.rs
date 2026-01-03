use std::path::Path;

use eyre::Result;

pub(crate) fn write_fixture_repo(root: &Path) -> Result<()> {
  std::fs::create_dir_all(root.join("src"))?;

  std::fs::write(
    root.join("src/lib.rs"),
    "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\
     pub fn run() -> i32 { add(1, 2) }\n",
  )?;

  std::fs::write(
    root.join("src/app.py"),
    "def add(a, b):\n    return a + b\n\n\
def run():\n    return add(1, 2)\n",
  )?;

  std::fs::write(
    root.join("src/app.go"),
    "package main\n\nfunc add(a int, b int) int { return a + b }\n\
func run() int { return add(1, 2) }\n",
  )?;

  std::fs::write(
    root.join("src/app.ts"),
    "export function add(a: number, b: number) { return a + b; }\n\
export function run() { return add(1, 2); }\n",
  )?;

  std::fs::write(
    root.join("src/app.js"),
    "function add(a, b) { return a + b; }\n\
function run() { return add(1, 2); }\n\
module.exports = { add, run };\n",
  )?;

  Ok(())
}
