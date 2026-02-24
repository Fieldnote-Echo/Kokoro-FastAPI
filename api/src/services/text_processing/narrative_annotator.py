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

# Markdown heading: 1-6 leading '#' followed by space and text
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")

# Dialogue: straight double-quoted or curly double-quoted text
_DIALOGUE_PATTERN = re.compile(
    r'("(?:[^"\\]|\\.)*"'
    r'|\u201c(?:[^\u201d\\]|\\.)*\u201d)',
)
# Aside: parenthetical text
_ASIDE_PATTERN = re.compile(r"(\([^)]+\))")


def annotate(text: str) -> list[NarrativeSegment]:
    """Analyze raw text and produce an annotated segment stream.

    Detection order: section breaks -> paragraph breaks -> headings -> dialogue -> asides.
    Each layer refines segments from the previous layer.
    """
    if not text or not text.strip():
        return []

    segments = _split_sections_and_paragraphs(text)
    segments = _detect_headings(segments)
    segments = _detect_dialogue_and_asides(segments)
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


def _detect_headings(segments: list[NarrativeSegment]) -> list[NarrativeSegment]:
    """Detect markdown headings and add before/after pause annotations.

    Heading text has the ``# `` prefix stripped and receives two annotations:
    a *before* annotation (with ``force_chunk_boundary=True``) and an *after*
    annotation for the trailing pause.
    """
    result: list[NarrativeSegment] = []
    for seg in segments:
        m = _HEADING_PATTERN.match(seg.text)
        if m:
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            # Preserve any existing annotations (e.g., paragraph_break from
            # a heading that follows another paragraph).
            annotations = list(seg.annotations)
            annotations.append(
                NarrativeAnnotation(
                    kind="heading",
                    pause_s=_settings.narrative_heading_before_pause,
                    position="before",
                    force_chunk_boundary=True,
                    metadata={"level": level},
                )
            )
            annotations.append(
                NarrativeAnnotation(
                    kind="heading",
                    pause_s=_settings.narrative_heading_after_pause,
                    position="after",
                    force_chunk_boundary=False,
                    metadata={"level": level},
                )
            )
            result.append(NarrativeSegment(text=heading_text, annotations=annotations))
        else:
            result.append(seg)
    return result


def _split_on_pattern(
    segments: list[NarrativeSegment],
    pattern: re.Pattern[str],
    kind: str,
    skip_kinds: tuple[str, ...] = ("heading",),
) -> list[NarrativeSegment]:
    """Split segments on *pattern*, annotating matched spans with *kind*.

    For each segment that is NOT annotated with a kind in *skip_kinds*, we
    search for *pattern* matches.  When a match is found mid-text the segment
    is split into up to three sub-segments: text before the match, the matched
    span (annotated), and text after the match.  The original segment's
    annotations are carried only on the first sub-segment.
    """
    result: list[NarrativeSegment] = []
    for seg in segments:
        # Skip segments already annotated with any of the skip_kinds.
        if any(a.kind in skip_kinds for a in seg.annotations):
            result.append(seg)
            continue

        text = seg.text
        parts: list[NarrativeSegment] = []
        last_end = 0
        first_sub = True

        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            # Text before the match
            before_text = text[last_end:start].strip()
            if before_text:
                anns = list(seg.annotations) if first_sub else []
                parts.append(NarrativeSegment(text=before_text, annotations=anns))
                first_sub = False

            # The matched span itself, annotated
            matched_text = m.group(0)
            match_anns = list(seg.annotations) if first_sub else []
            match_anns.append(
                NarrativeAnnotation(
                    kind=kind,
                    pause_s=0.0,
                    position="before",
                    force_chunk_boundary=True,
                )
            )
            parts.append(NarrativeSegment(text=matched_text, annotations=match_anns))
            first_sub = False
            last_end = end

        if not parts:
            # No matches — keep segment unchanged
            result.append(seg)
        else:
            # Text after last match
            remaining = text[last_end:].strip()
            if remaining:
                parts.append(NarrativeSegment(text=remaining, annotations=[]))
            result.extend(parts)

    return result


def _detect_dialogue_and_asides(
    segments: list[NarrativeSegment],
) -> list[NarrativeSegment]:
    """Detect dialogue (quoted text) and asides (parentheticals).

    Dialogue detection runs FIRST so that parentheticals inside quotes are
    preserved as part of the dialogue span rather than being split as asides.
    """
    segments = _split_on_pattern(segments, _DIALOGUE_PATTERN, "dialogue")
    segments = _split_on_pattern(
        segments, _ASIDE_PATTERN, "aside", skip_kinds=("heading", "dialogue")
    )
    return segments
