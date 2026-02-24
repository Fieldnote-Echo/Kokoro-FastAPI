"""Narrative awareness annotator for long-form TTS.

Detects document structure (paragraphs, sections, headings, dialogue, asides)
and produces an annotated segment stream that the TTS pipeline translates
into pauses and chunk boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

try:
    from ...core.config import settings as _settings
except ImportError:
    # Fallback for direct module loading (e.g., importlib in tests outside Docker)
    from api.src.core.config import settings as _settings


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


# Section break patterns (consumed as dividers, not spoken).
# Matches a line of 3+ dashes/asterisks/underscores surrounded by paragraph breaks.
_SECTION_BREAK_PATTERN = re.compile(
    r"\n*[\s]*[-*_]{3,}[\s]*\n*", re.MULTILINE
)
# Triple+ newlines are also section breaks
_TRIPLE_NEWLINE = re.compile(r"\n{3,}")

_SENTINEL = "\x00SECTION_BREAK\x00"


def annotate(text: str) -> list[NarrativeSegment]:
    """Analyze raw text and produce an annotated segment stream.

    Detection order: section breaks -> paragraph breaks -> headings -> dialogue -> asides.
    Each layer refines segments from the previous layer.
    """
    if not text or not text.strip():
        return []

    segments = _split_sections_and_paragraphs(text)
    return segments


def _split_sections_and_paragraphs(text: str) -> list[NarrativeSegment]:
    """Split text into segments at section and paragraph boundaries."""
    # Replace explicit section dividers (---, ***, ___) with sentinel
    processed = _SECTION_BREAK_PATTERN.sub(_SENTINEL, text)
    # Replace triple+ newlines with sentinel (also a section break)
    processed = _TRIPLE_NEWLINE.sub(_SENTINEL, processed)

    # Split on double newlines (paragraph boundaries)
    parts = re.split(r"\n\n+", processed)

    segments: list[NarrativeSegment] = []
    next_is_section_break = False

    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue

        # Check if this part contains a section break sentinel
        if _SENTINEL in stripped:
            # Split on sentinel to get text chunks between section breaks
            sub_parts = stripped.split(_SENTINEL)

            for i, sub_raw in enumerate(sub_parts):
                sub = sub_raw.strip()
                if not sub:
                    # Empty sub-part means a sentinel was at start/end/consecutive.
                    # The next non-empty sub-part should get a section_break annotation.
                    next_is_section_break = True
                    continue

                annotations: list[NarrativeAnnotation] = []
                if next_is_section_break:
                    annotations.append(
                        NarrativeAnnotation(
                            kind="section_break",
                            pause_s=_settings.narrative_section_pause,
                            position="before",
                            force_chunk_boundary=True,
                        )
                    )
                    next_is_section_break = False
                elif segments:
                    annotations.append(
                        NarrativeAnnotation(
                            kind="paragraph_break",
                            pause_s=_settings.narrative_paragraph_pause,
                            position="before",
                            force_chunk_boundary=False,
                        )
                    )
                segments.append(NarrativeSegment(text=sub, annotations=annotations))

                # After processing a sub-part, if there are more parts after it
                # (i.e., another sentinel follows), mark the next as section break.
                if i < len(sub_parts) - 1:
                    next_is_section_break = True
            continue

        annotations = []

        if next_is_section_break:
            annotations.append(
                NarrativeAnnotation(
                    kind="section_break",
                    pause_s=_settings.narrative_section_pause,
                    position="before",
                    force_chunk_boundary=True,
                )
            )
            next_is_section_break = False
        elif segments:
            annotations.append(
                NarrativeAnnotation(
                    kind="paragraph_break",
                    pause_s=_settings.narrative_paragraph_pause,
                    position="before",
                    force_chunk_boundary=False,
                )
            )

        segments.append(NarrativeSegment(text=stripped, annotations=annotations))

    return segments
