"""Narrative awareness annotator for long-form TTS.

Detects document structure (paragraphs, sections, headings, dialogue, asides)
and produces an annotated segment stream that the TTS pipeline translates
into pauses and chunk boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NarrativeAnnotation:
    """A structural annotation on a text segment."""

    kind: str  # "paragraph_break", "section_break", "heading", "dialogue", "aside"
    pause_s: float  # silence duration in seconds (0.0 = no pause, just chunk boundary)
    position: str  # "before" or "after" — when to emit this pause relative to segment text
    force_chunk_boundary: bool  # if True, this segment must start a new chunk
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NarrativeSegment:
    """A segment of text with optional structural annotations."""

    text: str
    annotations: list[NarrativeAnnotation] = field(default_factory=list)
