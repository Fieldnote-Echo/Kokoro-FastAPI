# Narrative Awareness MVP — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Kokoro's TTS preprocessing layer aware of document structure (paragraphs, sections, headings, dialogue, asides) so that long-form narration produces audible pacing at structural boundaries.

**Architecture:** A new `annotate()` function detects structural elements in raw text before normalization destroys them, producing `NarrativeSegment` objects internally. A `segments_to_tagged_text()` function converts these back to a string with `[pause:Xs]` tags injected at structural boundaries. This tagged string is passed to the existing `smart_split()` pipeline unchanged — zero modifications to the core chunking engine.

**Tech Stack:** Python dataclasses, regex pattern matching, existing `[pause:Xs]` infrastructure in `smart_split()`

**Design doc:** `docs/architecture/narrative-awareness-design.md`

---

## Task 1: Add Narrative Pause Settings to Config

**Files:**
- Modify: `api/src/core/config.py:35-53` (Text Processing Settings section)

**Step 1: Write the failing test**

Create `api/tests/test_narrative_annotator.py`:

```python
"""Tests for narrative awareness annotator."""

import pytest


def test_narrative_settings_exist():
    """Config includes narrative pause duration settings."""
    from api.src.core.config import Settings

    s = Settings()
    assert hasattr(s, "narrative_paragraph_pause")
    assert hasattr(s, "narrative_section_pause")
    assert hasattr(s, "narrative_heading_before_pause")
    assert hasattr(s, "narrative_heading_after_pause")
    assert s.narrative_paragraph_pause == 0.6
    assert s.narrative_section_pause == 1.2
    assert s.narrative_heading_before_pause == 1.0
    assert s.narrative_heading_after_pause == 0.8
```

**Step 2: Run test to verify it fails**

Run: `cd /home/ndspence/GitHub/Kokoro-FastAPI && python -m pytest api/tests/test_narrative_annotator.py::test_narrative_settings_exist -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'narrative_paragraph_pause'`

**Step 3: Add settings**

In `api/src/core/config.py`, after `advanced_text_normalization` (line 39), add:

```python
    # Narrative Awareness Settings
    narrative_paragraph_pause: float = 0.6     # Seconds of silence between paragraphs
    narrative_section_pause: float = 1.2       # Seconds of silence at section breaks
    narrative_heading_before_pause: float = 1.0  # Seconds of silence before headings
    narrative_heading_after_pause: float = 0.8   # Seconds of silence after headings
```

**Step 4: Run test to verify it passes**

Run: `cd /home/ndspence/GitHub/Kokoro-FastAPI && python -m pytest api/tests/test_narrative_annotator.py::test_narrative_settings_exist -v`
Expected: PASS

**Step 5: Commit**

```bash
git add api/src/core/config.py api/tests/test_narrative_annotator.py
git commit -m "feat(config): add narrative awareness pause duration settings"
```

---

## Task 2: Create NarrativeAnnotation and NarrativeSegment Dataclasses

**Files:**
- Create: `api/src/services/text_processing/narrative_annotator.py`
- Modify: `api/tests/test_narrative_annotator.py`

**Step 1: Write the failing test**

Add to `api/tests/test_narrative_annotator.py`:

```python
def test_narrative_segment_dataclass():
    """NarrativeSegment holds text with optional annotations."""
    from api.src.services.text_processing.narrative_annotator import (
        NarrativeAnnotation,
        NarrativeSegment,
    )

    seg = NarrativeSegment(text="Hello world.", annotations=[])
    assert seg.text == "Hello world."
    assert seg.annotations == []

    ann = NarrativeAnnotation(
        kind="paragraph_break",
        pause_s=0.6,
        position="before",
        force_chunk_boundary=False,
        metadata={},
    )
    seg_with_ann = NarrativeSegment(text="Next paragraph.", annotations=[ann])
    assert seg_with_ann.annotations[0].kind == "paragraph_break"
    assert seg_with_ann.annotations[0].pause_s == 0.6
    assert seg_with_ann.annotations[0].position == "before"


def test_heading_has_before_and_after_annotations():
    """Heading segments can carry both before and after annotations."""
    from api.src.services.text_processing.narrative_annotator import (
        NarrativeAnnotation,
        NarrativeSegment,
    )

    seg = NarrativeSegment(
        text="Chapter One",
        annotations=[
            NarrativeAnnotation(kind="heading", pause_s=1.0, position="before", force_chunk_boundary=True, metadata={"level": 1}),
            NarrativeAnnotation(kind="heading", pause_s=0.8, position="after", force_chunk_boundary=False, metadata={"level": 1}),
        ],
    )
    before = [a for a in seg.annotations if a.position == "before"]
    after = [a for a in seg.annotations if a.position == "after"]
    assert len(before) == 1
    assert len(after) == 1
    assert before[0].pause_s == 1.0
    assert after[0].pause_s == 0.8
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/ndspence/GitHub/Kokoro-FastAPI && python -m pytest api/tests/test_narrative_annotator.py -k "dataclass or heading_has" -v`
Expected: FAIL — `ImportError`

**Step 3: Create the annotator module with dataclasses**

Create `api/src/services/text_processing/narrative_annotator.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/ndspence/GitHub/Kokoro-FastAPI && python -m pytest api/tests/test_narrative_annotator.py -k "dataclass or heading_has" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add api/src/services/text_processing/narrative_annotator.py api/tests/test_narrative_annotator.py
git commit -m "feat(annotator): add NarrativeAnnotation and NarrativeSegment dataclasses"
```

---

## Task 3: Implement `annotate()` — Section and Paragraph Detection

**Files:**
- Modify: `api/src/services/text_processing/narrative_annotator.py`
- Modify: `api/tests/test_narrative_annotator.py`

**Step 1: Write the failing tests**

Add to `api/tests/test_narrative_annotator.py`:

```python
def test_annotate_single_paragraph():
    """Single paragraph produces one segment with no annotations."""
    from api.src.services.text_processing.narrative_annotator import annotate

    segments = annotate("Just a single paragraph.")
    assert len(segments) == 1
    assert segments[0].text == "Just a single paragraph."
    assert segments[0].annotations == []


def test_annotate_two_paragraphs():
    """Double newline produces paragraph_break annotation on second segment."""
    from api.src.services.text_processing.narrative_annotator import annotate

    segments = annotate("First paragraph.\n\nSecond paragraph.")
    assert len(segments) == 2
    assert segments[0].text == "First paragraph."
    assert segments[0].annotations == []
    assert segments[1].text == "Second paragraph."
    assert len(segments[1].annotations) == 1
    assert segments[1].annotations[0].kind == "paragraph_break"
    assert segments[1].annotations[0].position == "before"


def test_annotate_section_break_dashes():
    """Triple dashes produce section_break annotation."""
    from api.src.services.text_processing.narrative_annotator import annotate

    segments = annotate("Before section.\n\n---\n\nAfter section.")
    # Should produce: "Before section." (no ann), "After section." (section_break)
    # The "---" itself is consumed as a divider, not spoken
    text_segments = [s for s in segments if s.text.strip()]
    assert len(text_segments) == 2
    after = text_segments[1]
    section_anns = [a for a in after.annotations if a.kind == "section_break"]
    assert len(section_anns) == 1
    assert section_anns[0].pause_s >= 1.0


def test_annotate_section_break_asterisks():
    """Triple asterisks produce section_break annotation."""
    from api.src.services.text_processing.narrative_annotator import annotate

    segments = annotate("Before.\n\n***\n\nAfter.")
    text_segments = [s for s in segments if s.text.strip()]
    assert len(text_segments) == 2
    section_anns = [a for a in text_segments[1].annotations if a.kind == "section_break"]
    assert len(section_anns) == 1


def test_annotate_triple_newline_as_section_break():
    """Three or more newlines produce section_break instead of paragraph_break."""
    from api.src.services.text_processing.narrative_annotator import annotate

    segments = annotate("Before.\n\n\n\nAfter.")
    text_segments = [s for s in segments if s.text.strip()]
    assert len(text_segments) == 2
    ann = text_segments[1].annotations[0]
    assert ann.kind == "section_break"


def test_annotate_empty_input():
    """Empty input produces empty segment list."""
    from api.src.services.text_processing.narrative_annotator import annotate

    assert annotate("") == []
    assert annotate("   ") == []


def test_annotate_preserves_text_content():
    """Annotation doesn't modify or strip the text content within segments."""
    from api.src.services.text_processing.narrative_annotator import annotate

    segments = annotate("  Indented text.  \n\nSecond paragraph.")
    assert segments[0].text == "Indented text."
    assert segments[1].text == "Second paragraph."
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/ndspence/GitHub/Kokoro-FastAPI && python -m pytest api/tests/test_narrative_annotator.py -k "annotate_" -v`
Expected: FAIL — `ImportError: cannot import name 'annotate'`

**Step 3: Implement `annotate()` — section and paragraph detection**

Add to `api/src/services/text_processing/narrative_annotator.py`:

```python
from ..core.config import settings  # Adjust import path if needed

# Section break patterns (consumed as dividers, not spoken)
_SECTION_BREAK_PATTERN = re.compile(r"^[\s]*[-*_]{3,}[\s]*$", re.MULTILINE)
# Triple+ newlines are also section breaks
_TRIPLE_NEWLINE = re.compile(r"\n{3,}")


def annotate(text: str) -> list[NarrativeSegment]:
    """Analyze raw text and produce an annotated segment stream.

    Detection order: section breaks -> paragraph breaks -> headings -> dialogue -> asides.
    Each layer refines segments from the previous layer.

    Args:
        text: Raw input text, potentially with markdown formatting.

    Returns:
        List of NarrativeSegment objects with structural annotations.
    """
    if not text or not text.strip():
        return []

    segments = _split_sections_and_paragraphs(text)
    # Future: segments = _detect_headings(segments)
    # Future: segments = _detect_dialogue(segments)
    # Future: segments = _detect_asides(segments)
    return segments


def _split_sections_and_paragraphs(text: str) -> list[NarrativeSegment]:
    """Split text into segments at section and paragraph boundaries."""
    # First, normalize section break markers to a sentinel
    sentinel = "\n\n\x00SECTION_BREAK\x00\n\n"
    processed = _SECTION_BREAK_PATTERN.sub("\x00SECTION_BREAK\x00", text)
    # Triple+ newlines are also section breaks
    processed = _TRIPLE_NEWLINE.sub(sentinel, processed)

    # Split on double newlines (paragraph boundaries)
    parts = re.split(r"\n\n+", processed)

    segments: list[NarrativeSegment] = []
    next_is_section_break = False

    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue

        # Check if this part is a section break sentinel
        if "\x00SECTION_BREAK\x00" in stripped:
            next_is_section_break = True
            # If there's text around the sentinel, extract it
            text_parts = stripped.replace("\x00SECTION_BREAK\x00", "").strip()
            if text_parts:
                stripped = text_parts
            else:
                continue

        annotations: list[NarrativeAnnotation] = []

        if next_is_section_break:
            annotations.append(
                NarrativeAnnotation(
                    kind="section_break",
                    pause_s=settings.narrative_section_pause,
                    position="before",
                    force_chunk_boundary=True,
                )
            )
            next_is_section_break = False
        elif segments:  # Not the first segment — it's a paragraph break
            annotations.append(
                NarrativeAnnotation(
                    kind="paragraph_break",
                    pause_s=settings.narrative_paragraph_pause,
                    position="before",
                    force_chunk_boundary=False,
                )
            )

        segments.append(NarrativeSegment(text=stripped, annotations=annotations))

    return segments
```

Note: The import for `settings` may need adjustment depending on the module's position in the package. Check that `from ...core.config import settings` works from inside `api/src/services/text_processing/`. If not, use `from api.src.core.config import settings`.

**Step 4: Run tests to verify they pass**

Run: `cd /home/ndspence/GitHub/Kokoro-FastAPI && python -m pytest api/tests/test_narrative_annotator.py -k "annotate_" -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add api/src/services/text_processing/narrative_annotator.py api/tests/test_narrative_annotator.py
git commit -m "feat(annotator): implement section and paragraph break detection"
```

---

## Task 4: Implement `annotate()` — Heading Detection

**Files:**
- Modify: `api/src/services/text_processing/narrative_annotator.py`
- Modify: `api/tests/test_narrative_annotator.py`

**Step 1: Write the failing tests**

Add to `api/tests/test_narrative_annotator.py`:

```python
def test_annotate_heading():
    """Markdown heading produces heading annotations (before + after)."""
    from api.src.services.text_processing.narrative_annotator import annotate

    segments = annotate("# Chapter One\n\nThe story begins.")
    assert len(segments) == 2

    heading = segments[0]
    assert heading.text == "Chapter One"
    before_anns = [a for a in heading.annotations if a.position == "before"]
    after_anns = [a for a in heading.annotations if a.position == "after"]
    assert len(before_anns) == 1
    assert before_anns[0].kind == "heading"
    assert before_anns[0].force_chunk_boundary is True
    assert len(after_anns) == 1
    assert after_anns[0].kind == "heading"
    assert after_anns[0].pause_s > 0

    body = segments[1]
    assert body.text == "The story begins."


def test_annotate_heading_levels():
    """Different heading levels are captured in metadata."""
    from api.src.services.text_processing.narrative_annotator import annotate

    segments = annotate("## Section Title\n\nContent.")
    heading = segments[0]
    before = [a for a in heading.annotations if a.position == "before"][0]
    assert before.metadata.get("level") == 2


def test_annotate_heading_not_first_segment():
    """Heading after body text gets both heading + paragraph annotations."""
    from api.src.services.text_processing.narrative_annotator import annotate

    segments = annotate("Intro text.\n\n## Next Section\n\nMore text.")
    assert len(segments) == 3
    # Second segment is the heading
    heading = segments[1]
    assert heading.text == "Next Section"
    kinds = {a.kind for a in heading.annotations}
    assert "heading" in kinds
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/ndspence/GitHub/Kokoro-FastAPI && python -m pytest api/tests/test_narrative_annotator.py -k "heading" -v`
Expected: FAIL

**Step 3: Implement heading detection**

Add to `narrative_annotator.py`:

```python
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")


def _detect_headings(segments: list[NarrativeSegment]) -> list[NarrativeSegment]:
    """Detect markdown headings within segments and annotate them."""
    result: list[NarrativeSegment] = []

    for segment in segments:
        match = _HEADING_PATTERN.match(segment.text)
        if match:
            level = len(match.group(1))
            heading_text = match.group(2).strip()

            annotations = list(segment.annotations)  # preserve existing (e.g., paragraph_break)
            annotations.append(
                NarrativeAnnotation(
                    kind="heading",
                    pause_s=settings.narrative_heading_before_pause,
                    position="before",
                    force_chunk_boundary=True,
                    metadata={"level": level},
                )
            )
            annotations.append(
                NarrativeAnnotation(
                    kind="heading",
                    pause_s=settings.narrative_heading_after_pause,
                    position="after",
                    force_chunk_boundary=False,
                    metadata={"level": level},
                )
            )
            result.append(NarrativeSegment(text=heading_text, annotations=annotations))
        else:
            result.append(segment)

    return result
```

And uncomment the heading detection call in `annotate()`:

```python
    segments = _split_sections_and_paragraphs(text)
    segments = _detect_headings(segments)
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/ndspence/GitHub/Kokoro-FastAPI && python -m pytest api/tests/test_narrative_annotator.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add api/src/services/text_processing/narrative_annotator.py api/tests/test_narrative_annotator.py
git commit -m "feat(annotator): detect markdown headings with before/after pause annotations"
```

---

## Task 5: Implement `annotate()` — Dialogue and Aside Detection

**Files:**
- Modify: `api/src/services/text_processing/narrative_annotator.py`
- Modify: `api/tests/test_narrative_annotator.py`

**Step 1: Write the failing tests**

Add to `api/tests/test_narrative_annotator.py`:

```python
def test_annotate_dialogue_straight_quotes():
    """Double-quoted text is split into dialogue segment."""
    from api.src.services.text_processing.narrative_annotator import annotate

    segments = annotate('She said "hello" and left.')
    texts = [s.text for s in segments]
    assert any('"hello"' in t for t in texts)
    dialogue_segs = [s for s in segments if any(a.kind == "dialogue" for a in s.annotations)]
    assert len(dialogue_segs) == 1
    assert dialogue_segs[0].annotations[0].force_chunk_boundary is True


def test_annotate_dialogue_curly_quotes():
    """Curly-quoted text is split into dialogue segment."""
    from api.src.services.text_processing.narrative_annotator import annotate

    segments = annotate("She said \u201chello\u201d and left.")
    dialogue_segs = [s for s in segments if any(a.kind == "dialogue" for a in s.annotations)]
    assert len(dialogue_segs) == 1


def test_annotate_dialogue_full_paragraph():
    """Paragraph that is entirely dialogue gets one dialogue segment."""
    from api.src.services.text_processing.narrative_annotator import annotate

    segments = annotate('"Cold out there," she said.')
    assert len(segments) == 1
    # The whole thing is dialogue context — still one segment
    # (quotes span the full text minus attribution)


def test_annotate_aside_parenthetical():
    """Parenthetical text is split into aside segment."""
    from api.src.services.text_processing.narrative_annotator import annotate

    segments = annotate("He swept (as always) and oiled the bench.")
    aside_segs = [s for s in segments if any(a.kind == "aside" for a in s.annotations)]
    assert len(aside_segs) == 1
    assert "(as always)" in aside_segs[0].text


def test_annotate_dialogue_before_aside():
    """Dialogue detection runs before aside — parenthetical inside quotes stays in dialogue."""
    from api.src.services.text_processing.narrative_annotator import annotate

    segments = annotate('She said "I told him (the one from Tuesday) to leave" and walked away.')
    dialogue_segs = [s for s in segments if any(a.kind == "dialogue" for a in s.annotations)]
    assert len(dialogue_segs) == 1
    # The parenthetical should be inside the dialogue, not split out
    assert "(the one from Tuesday)" in dialogue_segs[0].text


def test_annotate_no_dialogue_in_plain_text():
    """Text without quotes produces no dialogue annotations."""
    from api.src.services.text_processing.narrative_annotator import annotate

    segments = annotate("The salt was heavy. She put it down.")
    dialogue_segs = [s for s in segments if any(a.kind == "dialogue" for a in s.annotations)]
    assert len(dialogue_segs) == 0
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/ndspence/GitHub/Kokoro-FastAPI && python -m pytest api/tests/test_narrative_annotator.py -k "dialogue or aside" -v`
Expected: FAIL

**Step 3: Implement dialogue and aside detection**

Add to `narrative_annotator.py`:

```python
# Dialogue: straight double quotes or curly double quotes
_DIALOGUE_PATTERN = re.compile(
    r'("(?:[^"\\]|\\.)*"'       # straight double quotes
    r'|\u201c(?:[^\u201d\\]|\\.)*\u201d)',  # curly double quotes
)

# Aside: parentheticals (but NOT inside dialogue — handled by detection order)
_ASIDE_PATTERN = re.compile(r"(\([^)]+\))")


def _detect_dialogue_and_asides(segments: list[NarrativeSegment]) -> list[NarrativeSegment]:
    """Detect dialogue and asides within segments, splitting them into sub-segments.

    Dialogue detection runs first so parentheticals inside quotes are preserved.
    """
    result: list[NarrativeSegment] = []

    for segment in segments:
        # Skip segments that already have heading annotations (don't split headings)
        if any(a.kind == "heading" for a in segment.annotations):
            result.append(segment)
            continue

        sub_segments = _split_on_pattern(segment, _DIALOGUE_PATTERN, "dialogue")
        final_segments: list[NarrativeSegment] = []
        for sub in sub_segments:
            # Only run aside detection on non-dialogue segments
            if any(a.kind == "dialogue" for a in sub.annotations):
                final_segments.append(sub)
            else:
                final_segments.extend(_split_on_pattern(sub, _ASIDE_PATTERN, "aside"))

        result.extend(final_segments)

    return result


def _split_on_pattern(
    segment: NarrativeSegment,
    pattern: re.Pattern,
    kind: str,
) -> list[NarrativeSegment]:
    """Split a segment's text on a regex pattern, creating annotated sub-segments."""
    parts = pattern.split(segment.text)

    if len(parts) == 1:
        return [segment]  # No matches, return unchanged

    result: list[NarrativeSegment] = []
    is_first = True

    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue

        if pattern.fullmatch(part):
            # This part matches the pattern — annotate it
            result.append(
                NarrativeSegment(
                    text=stripped,
                    annotations=[
                        NarrativeAnnotation(
                            kind=kind,
                            pause_s=0.0,
                            position="before",
                            force_chunk_boundary=True,
                        )
                    ],
                )
            )
        else:
            # Regular text — carry forward the original segment's annotations only on the first part
            annotations = list(segment.annotations) if is_first else []
            result.append(NarrativeSegment(text=stripped, annotations=annotations))

        is_first = False

    return result if result else [segment]
```

And uncomment the detection calls in `annotate()`:

```python
    segments = _split_sections_and_paragraphs(text)
    segments = _detect_headings(segments)
    segments = _detect_dialogue_and_asides(segments)
    return segments
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/ndspence/GitHub/Kokoro-FastAPI && python -m pytest api/tests/test_narrative_annotator.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add api/src/services/text_processing/narrative_annotator.py api/tests/test_narrative_annotator.py
git commit -m "feat(annotator): detect dialogue and asides with proper nesting order"
```

---

## Task 6: Implement `segments_to_tagged_text()` with Pause Collision Resolution

**Files:**
- Modify: `api/src/services/text_processing/narrative_annotator.py`
- Modify: `api/tests/test_narrative_annotator.py`

**Step 1: Write the failing tests**

Add to `api/tests/test_narrative_annotator.py`:

```python
def test_segments_to_tagged_text_simple():
    """Two paragraphs produce text with pause tag between them."""
    from api.src.services.text_processing.narrative_annotator import (
        annotate,
        segments_to_tagged_text,
    )

    segments = annotate("First paragraph.\n\nSecond paragraph.")
    tagged = segments_to_tagged_text(segments)
    assert "First paragraph." in tagged
    assert "[pause:" in tagged
    assert "Second paragraph." in tagged


def test_segments_to_tagged_text_section_break():
    """Section break produces longer pause than paragraph break."""
    from api.src.services.text_processing.narrative_annotator import (
        annotate,
        segments_to_tagged_text,
    )

    segments = annotate("Before.\n\n---\n\nAfter.")
    tagged = segments_to_tagged_text(segments)
    # Extract pause value
    import re
    pauses = re.findall(r"\[pause:([\d.]+)s\]", tagged)
    assert len(pauses) >= 1
    assert float(pauses[0]) >= 1.0  # section break pause


def test_segments_to_tagged_text_heading():
    """Heading gets pause before and after."""
    from api.src.services.text_processing.narrative_annotator import (
        annotate,
        segments_to_tagged_text,
    )

    segments = annotate("# Title\n\nBody text.")
    tagged = segments_to_tagged_text(segments)
    import re
    pauses = re.findall(r"\[pause:([\d.]+)s\]", tagged)
    assert len(pauses) >= 2  # before heading + after heading


def test_segments_to_tagged_text_collision_uses_max():
    """When section_break and heading coincide, max() pause wins."""
    from api.src.services.text_processing.narrative_annotator import (
        annotate,
        segments_to_tagged_text,
    )

    segments = annotate("Before.\n\n---\n\n# Heading\n\nAfter.")
    tagged = segments_to_tagged_text(segments)
    import re
    pauses = re.findall(r"\[pause:([\d.]+)s\]", tagged)
    # The heading has section_break (1.2) + heading_before (1.0) — max should be 1.2
    pause_values = [float(p) for p in pauses]
    assert max(pause_values) >= 1.2


def test_segments_to_tagged_text_no_leading_pause():
    """First segment doesn't get a leading pause (no silence before content starts)."""
    from api.src.services.text_processing.narrative_annotator import (
        annotate,
        segments_to_tagged_text,
    )

    segments = annotate("# Title\n\nBody.")
    tagged = segments_to_tagged_text(segments)
    # Should not start with a pause tag
    assert not tagged.lstrip().startswith("[pause:")


def test_segments_to_tagged_text_plain_text_unchanged():
    """Plain text without structure passes through unchanged."""
    from api.src.services.text_processing.narrative_annotator import (
        annotate,
        segments_to_tagged_text,
    )

    text = "Just a simple sentence with no structure."
    segments = annotate(text)
    tagged = segments_to_tagged_text(segments)
    assert tagged.strip() == text.strip()
    assert "[pause:" not in tagged
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/ndspence/GitHub/Kokoro-FastAPI && python -m pytest api/tests/test_narrative_annotator.py -k "tagged_text" -v`
Expected: FAIL — `ImportError`

**Step 3: Implement `segments_to_tagged_text()`**

Add to `narrative_annotator.py`:

```python
def segments_to_tagged_text(segments: list[NarrativeSegment]) -> str:
    """Convert annotated segments back to a string with [pause:Xs] tags.

    Pause collision policy: max() semantics — when multiple pauses coincide
    at the same boundary, the longest wins.

    Args:
        segments: Annotated segment list from annotate().

    Returns:
        String with [pause:Xs] tags injected at structural boundaries.
    """
    if not segments:
        return ""

    parts: list[str] = []
    is_first_text = True

    for segment in segments:
        if not segment.text.strip():
            continue

        # Collect "before" pauses
        before_pauses = [
            a.pause_s for a in segment.annotations
            if a.position == "before" and a.pause_s > 0
        ]
        has_chunk_boundary = any(
            a.force_chunk_boundary for a in segment.annotations if a.position == "before"
        )

        # Emit pause before this segment (skip for first text segment)
        if not is_first_text:
            if before_pauses:
                max_pause = max(before_pauses)
                parts.append(f" [pause:{max_pause}s] ")
            elif has_chunk_boundary:
                # Force chunk boundary with minimal pause
                parts.append(" [pause:0.001s] ")
            else:
                parts.append(" ")

        # Emit the text
        parts.append(segment.text)
        is_first_text = False

        # Collect "after" pauses
        after_pauses = [
            a.pause_s for a in segment.annotations
            if a.position == "after" and a.pause_s > 0
        ]
        if after_pauses:
            max_pause = max(after_pauses)
            parts.append(f" [pause:{max_pause}s] ")

    return "".join(parts).strip()
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/ndspence/GitHub/Kokoro-FastAPI && python -m pytest api/tests/test_narrative_annotator.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add api/src/services/text_processing/narrative_annotator.py api/tests/test_narrative_annotator.py
git commit -m "feat(annotator): implement segments_to_tagged_text with max() collision resolution"
```

---

## Task 7: Wire Into TTS Pipeline

**Files:**
- Modify: `api/src/services/tts_service.py:288-293`
- Modify: `api/src/services/text_processing/__init__.py`

**Step 1: Write the failing test**

Add to `api/tests/test_narrative_annotator.py`:

```python
@pytest.mark.asyncio
async def test_annotated_text_flows_through_smart_split():
    """Tagged text from annotator is compatible with smart_split's pause handling."""
    from api.src.services.text_processing.narrative_annotator import (
        annotate,
        segments_to_tagged_text,
    )
    from api.src.services.text_processing.text_processor import smart_split

    text = "First paragraph.\n\nSecond paragraph.\n\n---\n\nThird paragraph."
    segments = annotate(text)
    tagged = segments_to_tagged_text(segments)

    chunks = []
    async for chunk_text, tokens, pause_duration in smart_split(tagged):
        chunks.append((chunk_text, tokens, pause_duration))

    # Should have text chunks AND pause chunks
    text_chunks = [c for c in chunks if c[2] is None]
    pause_chunks = [c for c in chunks if c[2] is not None]
    assert len(text_chunks) >= 2  # at least "First paragraph" and one more
    assert len(pause_chunks) >= 1  # at least one structural pause
```

**Step 2: Run test to verify it passes** (this should already work since we're using existing infrastructure)

Run: `cd /home/ndspence/GitHub/Kokoro-FastAPI && python -m pytest api/tests/test_narrative_annotator.py::test_annotated_text_flows_through_smart_split -v`
Expected: PASS (the smart_split already handles `[pause:Xs]` tags)

**Step 3: Wire into tts_service.py**

In `api/src/services/tts_service.py`, add import at the top (near line 24):

```python
from .text_processing.narrative_annotator import annotate, segments_to_tagged_text
```

Then modify `generate_audio_stream()` around line 288. Change:

```python
            # Process text in chunks with smart splitting, handling pause tags
            async for chunk_text, tokens, pause_duration_s in smart_split(
                text,
                lang_code=pipeline_lang_code,
                normalization_options=normalization_options,
            ):
```

To:

```python
            # Annotate document structure before normalization destroys it
            narrative_segments = annotate(text)
            annotated_text = segments_to_tagged_text(narrative_segments)

            # Process text in chunks with smart splitting, handling pause tags
            async for chunk_text, tokens, pause_duration_s in smart_split(
                annotated_text,
                lang_code=pipeline_lang_code,
                normalization_options=normalization_options,
            ):
```

Also update `__init__.py` to export the new functions:

In `api/src/services/text_processing/__init__.py`, add:

```python
from .narrative_annotator import annotate, segments_to_tagged_text
```

And add to `__all__`:

```python
    "annotate",
    "segments_to_tagged_text",
```

**Step 4: Run the full existing test suite to verify no regressions**

Run: `cd /home/ndspence/GitHub/Kokoro-FastAPI && python -m pytest api/tests/ -v`
Expected: All existing tests pass

**Step 5: Commit**

```bash
git add api/src/services/tts_service.py api/src/services/text_processing/__init__.py api/tests/test_narrative_annotator.py
git commit -m "feat(tts): wire narrative annotator into generate_audio_stream pipeline"
```

---

## Task 8: Regression and Edge Case Tests

**Files:**
- Modify: `api/tests/test_narrative_annotator.py`

**Step 1: Write additional edge case tests**

Add to `api/tests/test_narrative_annotator.py`:

```python
def test_annotate_preserves_existing_pause_tags():
    """Explicit [pause:Xs] tags in input text pass through untouched."""
    from api.src.services.text_processing.narrative_annotator import (
        annotate,
        segments_to_tagged_text,
    )

    text = "Before pause. [pause:2.0s] After pause."
    segments = annotate(text)
    tagged = segments_to_tagged_text(segments)
    assert "[pause:2.0s]" in tagged


def test_annotate_all_headings_no_body():
    """Document of only headings doesn't crash."""
    from api.src.services.text_processing.narrative_annotator import annotate

    segments = annotate("# Title\n\n## Subtitle\n\n### Sub-subtitle")
    assert len(segments) == 3
    assert all(
        any(a.kind == "heading" for a in s.annotations)
        for s in segments
    )


def test_annotate_novel_excerpt():
    """Real prose from the novel annotates correctly."""
    from api.src.services.text_processing.narrative_annotator import (
        annotate,
        segments_to_tagged_text,
    )

    excerpt = (
        "Komako noticed the salt was wrong on a Tuesday.\n\n"
        "Not wrong in any way she could explain to her husband, "
        "who would have looked at her over his reading glasses and said "
        "\u201cit\u2019s salt, Komako,\u201d in that tone he used for things "
        "he considered beneath the dignity of language.\n\n"
        "She put the salt down and went about her morning."
    )

    segments = annotate(excerpt)
    tagged = segments_to_tagged_text(segments)

    # Should have paragraph pauses
    assert "[pause:" in tagged
    # Should have 3 text segments (3 paragraphs)
    text_segments = [s for s in segments if s.text.strip()]
    assert len(text_segments) == 3
    # Second paragraph should have dialogue detected
    dialogue_segs = [s for s in segments if any(a.kind == "dialogue" for a in s.annotations)]
    assert len(dialogue_segs) >= 1
```

**Step 2: Run all tests**

Run: `cd /home/ndspence/GitHub/Kokoro-FastAPI && python -m pytest api/tests/test_narrative_annotator.py -v`
Expected: All pass

**Step 3: Run the full test suite for regressions**

Run: `cd /home/ndspence/GitHub/Kokoro-FastAPI && python -m pytest api/tests/ -v`
Expected: All pass, no regressions

**Step 4: Commit**

```bash
git add api/tests/test_narrative_annotator.py
git commit -m "test(annotator): add edge case and novel excerpt regression tests"
```

---

## Task Dependency Graph

```
1 (config) → 2 (dataclasses) → 3 (section/paragraph) → 4 (headings) → 5 (dialogue/asides) → 6 (tagged text) → 7 (pipeline wiring) → 8 (edge cases)
```

All tasks are sequential — each depends on the previous.

## Estimated Impact

- **New files**: 2 (`narrative_annotator.py`, `test_narrative_annotator.py`)
- **Modified files**: 3 (`config.py`, `tts_service.py`, `__init__.py`)
- **New tests**: ~25
- **Commits**: 8
- **smart_split() changes**: Zero
- **normalizer.py changes**: Zero
