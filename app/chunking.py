"""Simple character-based chunking with overlap.

Phase 0 keeps this naive on purpose — splitting on character count rather
than sentence/semantic boundaries. This is a known weak point we can
revisit in Phase 2 (eval harness) once we can *measure* whether smarter
chunking actually improves retrieval quality, rather than assuming it
does.
"""

from dataclasses import dataclass
from pathlib import Path

from app.config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS


@dataclass
class Chunk:
    text: str
    source: str  # filename
    chunk_index: int  # position within the source file


def chunk_text(text: str, source: str) -> list[Chunk]:
    """Split text into overlapping chunks of CHUNK_SIZE_CHARS characters."""
    chunks: list[Chunk] = []
    start = 0
    index = 0
    text_len = len(text)

    if text_len == 0:
        return chunks

    while start < text_len:
        end = min(start + CHUNK_SIZE_CHARS, text_len)
        chunk_text_value = text[start:end].strip()

        if chunk_text_value:
            chunks.append(Chunk(text=chunk_text_value, source=source, chunk_index=index))
            index += 1

        if end == text_len:
            break

        start = end - CHUNK_OVERLAP_CHARS

    return chunks


def load_and_chunk_notes(notes_dir: str) -> list[Chunk]:
    """Load all .md files from notes_dir and chunk each one."""
    all_chunks: list[Chunk] = []
    notes_path = Path(notes_dir)

    for md_file in sorted(notes_path.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        all_chunks.extend(chunk_text(text, source=md_file.name))

    return all_chunks
