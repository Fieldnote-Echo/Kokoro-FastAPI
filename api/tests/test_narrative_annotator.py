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
