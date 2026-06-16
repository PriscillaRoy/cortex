"""Tool: list_notes

Lists available notes, optionally filtered by a topic keyword.
Returns metadata (filename, tags, date from frontmatter) without
the full note content — letting the agent discover what exists
before deciding which notes to read in full via summarize_note.

This is the "discovery" step in a multi-step agent chain:
  1. list_notes(filter="rl")          -> finds rl_basics.md
  2. summarize_note("rl_basics.md")   -> reads it in full
  3. search_notes("PPO vs Q-learning") -> finds specific chunks
  4. synthesize                        -> final answer

Without list_notes, the agent would have to guess note filenames
or rely entirely on semantic search — which may miss a note if
the query terms don't match the embedding well.
"""

import re
from pathlib import Path

from app.config import NOTES_DIR


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "list_notes",
        "description": (
            "List available notes, optionally filtered by a topic keyword. "
            "Returns note filenames, tags, and dates — not full content. "
            "Use this to discover which notes exist before reading them. "
            "Call this when the user asks 'what do I have about X' or when "
            "you need to find which note to read in full."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filter": {
                    "type": "string",
                    "description": (
                        "Optional keyword to filter notes by. Matches against "
                        "filename, tags, and title. Leave empty to list all notes."
                    ),
                },
            },
            "required": [],
        },
    },
}


def _parse_frontmatter(text: str) -> dict:
    """Extract YAML-style frontmatter fields from a note.

    Looks for lines like:
      Tags: observability, grafana, promql
      Date: 2026-04-28
      Title: PromQL Patterns
    """
    meta = {}
    for line in text[:500].split("\n"):
        line = line.strip()
        for field in ("tags", "date", "title"):
            if line.lower().startswith(field + ":"):
                meta[field] = line.split(":", 1)[1].strip()
    return meta


def run(filter: str = "") -> dict:
    """List notes, optionally filtered by keyword.

    Returns:
        {
            "notes": [{"filename": ..., "title": ..., "tags": ..., "date": ...}, ...],
            "count": int,
            "filter": str
        }
    """
    notes_dir = Path(NOTES_DIR)
    if not notes_dir.exists():
        return {"notes": [], "count": 0, "filter": filter, "error": "Notes directory not found"}

    results = []
    filter_lower = filter.lower().strip()

    for path in sorted(notes_dir.glob("*.md")):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue

        meta = _parse_frontmatter(content)
        filename = path.name

        # Match filter against filename, tags, title
        if filter_lower:
            searchable = " ".join([
                filename.lower(),
                meta.get("tags", "").lower(),
                meta.get("title", "").lower(),
            ])
            if filter_lower not in searchable:
                continue

        results.append({
            "filename": filename,
            "title": meta.get("title", filename.replace(".md", "").replace("_", " ").title()),
            "tags": meta.get("tags", ""),
            "date": meta.get("date", ""),
        })

    return {
        "notes": results,
        "count": len(results),
        "filter": filter,
    }


def format_for_llm(result: dict) -> str:
    """Format the tool result as text for the LLM's next context window."""
    if result.get("error"):
        return f"Error: {result['error']}"

    if not result["notes"]:
        f_str = f" matching {result['filter']!r}" if result["filter"] else ""
        return f"No notes found{f_str}."

    f_str = f" matching {result['filter']!r}" if result["filter"] else ""
    lines = [f"Found {result['count']} note(s){f_str}:"]
    for note in result["notes"]:
        line = f"  - {note['filename']}"
        if note["title"] and note["title"] != note["filename"]:
            line += f" ({note['title']})"
        if note["tags"]:
            line += f"  [tags: {note['tags']}]"
        if note["date"]:
            line += f"  [{note['date']}]"
        lines.append(line)

    return "\n".join(lines)
