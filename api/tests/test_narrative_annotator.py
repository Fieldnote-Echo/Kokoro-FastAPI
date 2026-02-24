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
