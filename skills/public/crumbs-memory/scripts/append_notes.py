#!/usr/bin/env python3
import argparse
import datetime as dt
import subprocess
import sys
from typing import Optional


SECTION_END_MARKERS = [
    "\ncomments:\n",
    "\ncreated_at:",
    "\nupdated_at:",
    "\nclosed_at:",
]


def run(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout


def extract_notes(issue_text: str) -> tuple[str, Optional[tuple[int, int]]]:
    marker = "\nnotes:\n"
    start = issue_text.find(marker)
    if start == -1:
        return "", None

    content_start = start + len(marker)
    end_candidates: list[int] = []
    for end_marker in SECTION_END_MARKERS:
        idx = issue_text.find(end_marker, content_start)
        if idx != -1:
            end_candidates.append(idx)
    content_end = min(end_candidates) if end_candidates else len(issue_text)
    return issue_text[content_start:content_end].rstrip("\n"), (content_start, content_end)


def build_entry(text: str, title: str, author: str, utc: bool) -> str:
    now = dt.datetime.now(dt.timezone.utc if utc else None)
    stamp = now.isoformat(timespec="seconds")
    author_suffix = f" — {author}" if author else ""
    return f"### {title} ({stamp}{author_suffix})\n\n{text.strip()}\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append a dated entry to a crumbs issue 'notes' field."
    )
    parser.add_argument("id", help="Issue ID (or prefix)")
    parser.add_argument(
        "--title",
        default="Session note",
        help="Entry title to add (default: %(default)s)",
    )
    parser.add_argument("--author", default="", help="Author name to include in entry header")
    parser.add_argument(
        "--utc",
        action="store_true",
        help="Use UTC timestamps (default: local time)",
    )
    parser.add_argument(
        "--text",
        default="",
        help="Entry body text (if omitted, read from stdin)",
    )
    args = parser.parse_args()

    entry_text = args.text
    if not entry_text:
        if sys.stdin.isatty():
            parser.error("Provide --text or pipe content on stdin.")
        entry_text = sys.stdin.read()

    issue_text = run(["crumbs", "issue", "get", args.id])
    existing_notes, _span = extract_notes(issue_text)

    entry = build_entry(entry_text, title=args.title, author=args.author, utc=args.utc)
    if existing_notes.strip():
        new_notes = f"{existing_notes.rstrip()}\n\n{entry}"
    else:
        new_notes = entry

    subprocess.run(
        ["crumbs", "issue", "update", args.id, "--notes", new_notes],
        check=True,
        text=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

