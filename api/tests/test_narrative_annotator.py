"""Tests for narrative awareness annotator.

Uses importlib to load the module directly, bypassing the services/__init__.py
import chain that requires Docker-only dependencies (kokoro, loguru, etc.).
"""

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_narrative_annotator():
    """Load narrative_annotator module directly by file path.

    This avoids triggering api/src/services/__init__.py which imports
    kokoro (only available inside the Docker container).
    """
    module_name = "api.src.services.text_processing.narrative_annotator"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "services"
        / "text_processing"
        / "narrative_annotator.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Task 1: Config settings
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Task 2: Dataclasses
# ---------------------------------------------------------------------------


def test_narrative_segment_dataclass():
    """NarrativeSegment holds text with optional annotations."""
    mod = _load_narrative_annotator()
    NarrativeAnnotation = mod.NarrativeAnnotation
    NarrativeSegment = mod.NarrativeSegment

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
    mod = _load_narrative_annotator()
    NarrativeAnnotation = mod.NarrativeAnnotation
    NarrativeSegment = mod.NarrativeSegment

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


# ---------------------------------------------------------------------------
# Task 3: annotate() — section and paragraph detection
# ---------------------------------------------------------------------------


def test_annotate_single_paragraph():
    """Single paragraph produces one segment with no annotations."""
    mod = _load_narrative_annotator()
    annotate = mod.annotate

    segments = annotate("Just a single paragraph.")
    assert len(segments) == 1
    assert segments[0].text == "Just a single paragraph."
    assert segments[0].annotations == []


def test_annotate_two_paragraphs():
    """Double newline produces paragraph_break annotation on second segment."""
    mod = _load_narrative_annotator()
    annotate = mod.annotate

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
    mod = _load_narrative_annotator()
    annotate = mod.annotate

    segments = annotate("Before section.\n\n---\n\nAfter section.")
    text_segments = [s for s in segments if s.text.strip()]
    assert len(text_segments) == 2
    after = text_segments[1]
    section_anns = [a for a in after.annotations if a.kind == "section_break"]
    assert len(section_anns) == 1
    assert section_anns[0].pause_s >= 1.0


def test_annotate_section_break_asterisks():
    """Triple asterisks produce section_break annotation."""
    mod = _load_narrative_annotator()
    annotate = mod.annotate

    segments = annotate("Before.\n\n***\n\nAfter.")
    text_segments = [s for s in segments if s.text.strip()]
    assert len(text_segments) == 2
    section_anns = [a for a in text_segments[1].annotations if a.kind == "section_break"]
    assert len(section_anns) == 1


def test_annotate_triple_newline_as_section_break():
    """Three or more newlines produce section_break instead of paragraph_break."""
    mod = _load_narrative_annotator()
    annotate = mod.annotate

    segments = annotate("Before.\n\n\n\nAfter.")
    text_segments = [s for s in segments if s.text.strip()]
    assert len(text_segments) == 2
    ann = text_segments[1].annotations[0]
    assert ann.kind == "section_break"


def test_annotate_empty_input():
    """Empty input produces empty segment list."""
    mod = _load_narrative_annotator()
    annotate = mod.annotate

    assert annotate("") == []
    assert annotate("   ") == []


def test_annotate_preserves_text_content():
    """Annotation strips leading/trailing whitespace from segments."""
    mod = _load_narrative_annotator()
    annotate = mod.annotate

    segments = annotate("  Indented text.  \n\nSecond paragraph.")
    assert segments[0].text == "Indented text."
    assert segments[1].text == "Second paragraph."


# ---------------------------------------------------------------------------
# Task 4: annotate() — heading detection
# ---------------------------------------------------------------------------


def test_annotate_heading():
    """Markdown heading produces heading annotations (before + after)."""
    mod = _load_narrative_annotator()
    annotate = mod.annotate

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
    mod = _load_narrative_annotator()
    annotate = mod.annotate

    segments = annotate("## Section Title\n\nContent.")
    heading = segments[0]
    before = [a for a in heading.annotations if a.position == "before"][0]
    assert before.metadata.get("level") == 2


def test_annotate_heading_not_first_segment():
    """Heading after body text gets both heading + paragraph annotations."""
    mod = _load_narrative_annotator()
    annotate = mod.annotate

    segments = annotate("Intro text.\n\n## Next Section\n\nMore text.")
    assert len(segments) == 3
    heading = segments[1]
    assert heading.text == "Next Section"
    kinds = {a.kind for a in heading.annotations}
    assert "heading" in kinds


# ---------------------------------------------------------------------------
# Task 5: annotate() — dialogue and aside detection
# ---------------------------------------------------------------------------


def test_annotate_dialogue_straight_quotes():
    """Double-quoted text is split into dialogue segment."""
    mod = _load_narrative_annotator()
    annotate = mod.annotate

    segments = annotate('She said "hello" and left.')
    texts = [s.text for s in segments]
    assert any('"hello"' in t for t in texts)
    dialogue_segs = [s for s in segments if any(a.kind == "dialogue" for a in s.annotations)]
    assert len(dialogue_segs) == 1
    assert dialogue_segs[0].annotations[-1].force_chunk_boundary is True


def test_annotate_dialogue_curly_quotes():
    """Curly-quoted text is split into dialogue segment."""
    mod = _load_narrative_annotator()
    annotate = mod.annotate

    segments = annotate("She said \u201chello\u201d and left.")
    dialogue_segs = [s for s in segments if any(a.kind == "dialogue" for a in s.annotations)]
    assert len(dialogue_segs) == 1


def test_annotate_aside_parenthetical():
    """Parenthetical text is split into aside segment."""
    mod = _load_narrative_annotator()
    annotate = mod.annotate

    segments = annotate("He swept (as always) and oiled the bench.")
    aside_segs = [s for s in segments if any(a.kind == "aside" for a in s.annotations)]
    assert len(aside_segs) == 1
    assert "(as always)" in aside_segs[0].text


def test_annotate_dialogue_before_aside():
    """Dialogue detection runs before aside — parenthetical inside quotes stays in dialogue."""
    mod = _load_narrative_annotator()
    annotate = mod.annotate

    segments = annotate('She said "I told him (the one from Tuesday) to leave" and walked away.')
    dialogue_segs = [s for s in segments if any(a.kind == "dialogue" for a in s.annotations)]
    assert len(dialogue_segs) == 1
    assert "(the one from Tuesday)" in dialogue_segs[0].text


def test_annotate_no_dialogue_in_plain_text():
    """Text without quotes produces no dialogue annotations."""
    mod = _load_narrative_annotator()
    annotate = mod.annotate

    segments = annotate("The salt was heavy. She put it down.")
    dialogue_segs = [s for s in segments if any(a.kind == "dialogue" for a in s.annotations)]
    assert len(dialogue_segs) == 0


# ---------------------------------------------------------------------------
# Task 6: segments_to_tagged_text()
# ---------------------------------------------------------------------------


def test_segments_to_tagged_text_simple():
    """Two paragraphs produce text with pause tag between them."""
    mod = _load_narrative_annotator()
    annotate = mod.annotate
    segments_to_tagged_text = mod.segments_to_tagged_text

    segments = annotate("First paragraph.\n\nSecond paragraph.")
    tagged = segments_to_tagged_text(segments)
    assert "First paragraph." in tagged
    assert "[pause:" in tagged
    assert "Second paragraph." in tagged


def test_segments_to_tagged_text_section_break():
    """Section break produces longer pause than paragraph break."""
    mod = _load_narrative_annotator()
    annotate = mod.annotate
    segments_to_tagged_text = mod.segments_to_tagged_text

    segments = annotate("Before.\n\n---\n\nAfter.")
    tagged = segments_to_tagged_text(segments)
    import re as _re
    pauses = _re.findall(r"\[pause:([\d.]+)s\]", tagged)
    assert len(pauses) >= 1
    assert float(pauses[0]) >= 1.0


def test_segments_to_tagged_text_heading():
    """Heading gets pause before and after."""
    mod = _load_narrative_annotator()
    annotate = mod.annotate
    segments_to_tagged_text = mod.segments_to_tagged_text

    segments = annotate("# Title\n\nBody text.")
    tagged = segments_to_tagged_text(segments)
    import re as _re
    pauses = _re.findall(r"\[pause:([\d.]+)s\]", tagged)
    assert len(pauses) >= 2


def test_segments_to_tagged_text_no_leading_pause():
    """First segment doesn't get a leading pause."""
    mod = _load_narrative_annotator()
    annotate = mod.annotate
    segments_to_tagged_text = mod.segments_to_tagged_text

    segments = annotate("# Title\n\nBody.")
    tagged = segments_to_tagged_text(segments)
    assert not tagged.lstrip().startswith("[pause:")


def test_segments_to_tagged_text_plain_text_unchanged():
    """Plain text without structure passes through unchanged."""
    mod = _load_narrative_annotator()
    annotate = mod.annotate
    segments_to_tagged_text = mod.segments_to_tagged_text

    text = "Just a simple sentence with no structure."
    segments = annotate(text)
    tagged = segments_to_tagged_text(segments)
    assert tagged.strip() == text.strip()
    assert "[pause:" not in tagged
