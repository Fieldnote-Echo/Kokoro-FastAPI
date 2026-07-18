"""Unit tests for handle_markdown() — the navi-os fork of Kokoro's normalizer."""

import pytest

from api.src.services.text_processing.normalizer import handle_markdown
from api.src.services.text_processing.text_processor import smart_split
from api.src.structures.schemas import NormalizationOptions

# ── Tests ────────────────────────────────────────────────────────────────


def test_bold():
    assert handle_markdown("**bold text**") == "bold text"


def test_italic_star():
    assert handle_markdown("*italic*") == "italic"


def test_italic_underscore():
    assert handle_markdown("_italic_") == "italic"


def test_snake_case_preserved():
    assert handle_markdown("snake_case_var") == "snake_case_var"


def test_strikethrough():
    assert handle_markdown("~~deleted~~") == "deleted"


def test_inline_code():
    assert handle_markdown("`code`") == "code"


def test_fenced_code_block():
    result = handle_markdown("```python\nprint(\"hi\")\n```")
    assert result.strip() == 'print("hi")'


def test_link():
    assert handle_markdown("[click here](https://example.com)") == "click here"


def test_image():
    assert handle_markdown("![alt text](image.png)") == "alt text"


def test_heading():
    assert handle_markdown("### My Heading") == "My Heading"


def test_blockquote():
    assert handle_markdown("> quoted text") == "quoted text"


def test_horizontal_rule():
    assert handle_markdown("---").strip() == ""


def test_unordered_list():
    assert handle_markdown("- item one") == "item one"


def test_ordered_list():
    assert handle_markdown("1. first item") == "first item"


def test_table():
    md = "| Name | Age |\n| --- | --- |\n| Alice | 30 |"
    result = handle_markdown(md)
    # Separator row removed; pipes replaced with spaces on table lines
    assert "Name" in result
    assert "Age" in result
    assert "Alice" in result
    assert "30" in result
    # No pipe characters should remain
    assert "|" not in result
    # Separator row content should be gone
    assert "---" not in result


def test_non_table_pipe_preserved():
    assert handle_markdown("true | false") == "true | false"


def test_nested_bold_italic():
    assert handle_markdown("***bold italic***") == "bold italic"


def test_code_block_preserves_markdown():
    """Markdown inside fenced code blocks must NOT be stripped."""
    result = handle_markdown("```\n**not stripped**\n```")
    assert "**not stripped**" in result


def test_inline_code_preserves_markdown():
    """Markdown inside inline code must NOT be stripped."""
    result = handle_markdown("`**bold**`")
    assert result.strip() == "**bold**"


def test_mixed_llm_output():
    """Integration test: a realistic LLM response with mixed markdown."""
    md = (
        "### Summary\n\n"
        "This is **important** information about *machine learning*.\n\n"
        "Key points:\n"
        "- First, read the [documentation](https://docs.example.com)\n"
        "- Then, run `pip install torch`\n\n"
        "Here is an example:\n\n"
        "```python\nmodel.train()\n```\n\n"
        "> Note: this is experimental\n\n"
        "| Metric | Value |\n"
        "| --- | --- |\n"
        "| Accuracy | 95% |\n"
    )
    result = handle_markdown(md)
    # Formatting artifacts should be gone
    assert "**" not in result
    assert "###" not in result
    assert "](http" not in result
    assert "| --- |" not in result
    # Content should remain
    assert "important" in result
    assert "machine learning" in result
    assert "documentation" in result
    assert "pip install torch" in result
    assert "model.train()" in result
    assert "experimental" in result
    assert "Accuracy" in result
    assert "95%" in result


def test_multiple_code_blocks():
    """Multiple fenced code blocks should all be protected."""
    md = "```\n**block1**\n```\nSome **bold** text\n```\n*block2*\n```"
    result = handle_markdown(md)
    assert "**block1**" in result
    assert "*block2*" in result
    assert "bold" in result
    assert "**bold**" not in result


def test_heading_levels():
    """All heading levels (1-6) should be stripped."""
    for i in range(1, 7):
        hashes = "#" * i
        assert handle_markdown(f"{hashes} Heading {i}") == f"Heading {i}"


def test_unordered_list_markers():
    """All unordered list markers (-, *, +) should be stripped."""
    assert handle_markdown("- dash item") == "dash item"
    # * is also an italic marker, but at start of line with space it's a list
    assert handle_markdown("+ plus item") == "plus item"


def test_bold_with_surrounding_text():
    assert handle_markdown("This is **bold** in context") == "This is bold in context"


def test_link_with_surrounding_text():
    result = handle_markdown("Click [here](https://x.com) now")
    assert result == "Click here now"


# ── Multiline emphasis (soft wraps) ──────────────────────────────────────


def test_bold_across_soft_wrap():
    """Emphasis should span a single newline (soft-wrapped paragraph)."""
    assert handle_markdown("**bold\ntext**") == "bold\ntext"


def test_italic_star_across_soft_wrap():
    assert handle_markdown("*ital\nic*") == "ital\nic"


def test_italic_underscore_across_soft_wrap():
    assert handle_markdown("_ital\nic_") == "ital\nic"


def test_strikethrough_across_soft_wrap():
    assert handle_markdown("~~gone\nnow~~") == "gone\nnow"


def test_emphasis_does_not_span_blank_lines():
    """A blank line is a paragraph break — emphasis must not match across it."""
    md = "**alpha\n\nbeta**"
    assert handle_markdown(md) == md


def test_emphasis_does_not_span_whitespace_only_blank_lines():
    md = "**alpha\n   \nbeta**"
    assert handle_markdown(md) == md


# ── Unclosed fenced code blocks (streaming / truncated chunks) ───────────


def test_unclosed_fenced_code_block():
    """An unclosed fence extends to end of input: fence line stripped, content kept."""
    result = handle_markdown("```python\nprint('hi')")
    assert result == "print('hi')"


def test_unclosed_fenced_code_block_with_prefix():
    result = handle_markdown("Here is code:\n```python\nmodel.train()\n")
    assert "`" not in result
    assert "python" not in result
    assert "model.train()" in result
    assert "Here is code:" in result


def test_unclosed_fence_protects_content():
    """Markdown inside an unclosed fence must not be stripped."""
    result = handle_markdown("```\n**not stripped**")
    assert "**not stripped**" in result


def test_bare_unclosed_fence_line():
    """A lone opening fence with no content voices nothing."""
    assert handle_markdown("```python").strip() == ""


# ── Parenthesized URLs in links and images ───────────────────────────────


def test_link_with_parenthesized_url():
    md = "[Foo](https://en.wikipedia.org/wiki/Foo_(bar))"
    assert handle_markdown(md) == "Foo"


def test_image_with_parenthesized_url():
    md = "![alt text](https://example.com/img_(1).png)"
    assert handle_markdown(md) == "alt text"


def test_link_paren_url_with_surrounding_text():
    md = "See [Foo](https://en.wikipedia.org/wiki/Foo_(bar)) today"
    assert handle_markdown(md) == "See Foo today"


def test_link_plain_url_still_stops_at_paren():
    """A plain link followed by a parenthetical must not over-match."""
    md = "[here](https://x.com) (see note)"
    assert handle_markdown(md) == "here (see note)"


# ── Custom phoneme spacing through smart_split ───────────────────────────


@pytest.mark.asyncio
async def test_custom_phoneme_spacing_preserved():
    """Words must not get glued to custom phoneme tokens during normalization.

    smart_split splits on CUSTOM_PHONEMES and rejoins segments with ''.join;
    if segment edges get stripped, 'Say [Kokoro](...) please' becomes
    'Say[Kokoro](...)please'.
    """
    chunks = []
    async for chunk_text, _, _ in smart_split(
        "Say [Kokoro](/kˈOkəɹO/) please.",
        normalization_options=NormalizationOptions(),
    ):
        chunks.append(chunk_text)

    combined = " ".join(chunks)
    assert "Say [Kokoro](/kˈOkəɹO/) please." in combined
    assert "Say[" not in combined
    assert ")please" not in combined


@pytest.mark.asyncio
async def test_custom_phoneme_spacing_preserved_between_tokens():
    """A whitespace-only segment between two phoneme tokens must survive."""
    chunks = []
    async for chunk_text, _, _ in smart_split(
        "[Kokoro](/kˈOkəɹO/) [rest](/ɹˈɛst/) now.",
        normalization_options=NormalizationOptions(),
    ):
        chunks.append(chunk_text)

    combined = " ".join(chunks)
    assert "/) [rest]" in combined
    assert "/)[rest]" not in combined


# ── Markdown normalization for non-English languages ─────────────────────


@pytest.mark.asyncio
async def test_markdown_stripped_for_non_english_language():
    """Markdown syntax is language-independent — it must be stripped even
    when the rest of (English-specific) normalization is skipped."""
    chunks = []
    async for chunk_text, _, _ in smart_split(
        "# Heading\nDas ist **fett** und [Link](https://example.com).",
        lang_code="e",
        normalization_options=NormalizationOptions(),
    ):
        chunks.append(chunk_text)

    combined = " ".join(chunks)
    assert "**" not in combined
    assert "#" not in combined
    assert "](" not in combined
    assert "fett" in combined
    assert "Link" in combined


@pytest.mark.asyncio
async def test_non_english_markdown_keeps_custom_phonemes():
    """Custom phoneme tokens look like markdown links — they must survive
    the markdown pass for non-English languages too."""
    chunks = []
    async for chunk_text, _, _ in smart_split(
        "Sag [Kokoro](/kˈOkəɹO/) **bitte**.",
        lang_code="e",
        normalization_options=NormalizationOptions(),
    ):
        chunks.append(chunk_text)

    combined = " ".join(chunks)
    assert "[Kokoro](/kˈOkəɹO/)" in combined
    assert "**" not in combined


@pytest.mark.asyncio
async def test_non_english_markdown_respects_toggle():
    """With markdown_normalization disabled, non-English text is untouched."""
    chunks = []
    async for chunk_text, _, _ in smart_split(
        "Das ist **fett**.",
        lang_code="e",
        normalization_options=NormalizationOptions(markdown_normalization=False),
    ):
        chunks.append(chunk_text)

    combined = " ".join(chunks)
    assert "**fett**" in combined


# ── Minor batch: emphasis, headings, links, sentinels, tables ────────────


def test_bold_double_underscore():
    assert handle_markdown("__bold__") == "bold"


def test_bold_double_underscore_with_surrounding_text():
    assert handle_markdown("This is __bold__ text") == "This is bold text"


def test_closed_atx_heading():
    """ATX headings may close with trailing hashes: '## H ##' -> 'H'."""
    assert handle_markdown("## Heading ##") == "Heading"


def test_heading_with_trailing_hash_word_kept():
    """A '#' glued to a word is content, not a closing sequence."""
    assert handle_markdown("# Learning C#") == "Learning C#"


def test_heading_after_list_marker():
    """List markers are stripped before headings so '- # H' voices 'H'."""
    assert handle_markdown("- # Heading") == "Heading"


def test_heading_after_ordered_list_marker():
    assert handle_markdown("1. ## Heading") == "Heading"


def test_empty_link_text_dropped():
    result = handle_markdown("before [](https://example.com) after")
    assert "[" not in result
    assert "]" not in result
    assert "example.com" not in result
    assert "before" in result
    assert "after" in result


def test_sentinel_collision_safe():
    """Literal placeholder-looking input must not corrupt code restoration."""
    md = "weird \x00CB0\x00 input\n```\nsecret code\n```"
    result = handle_markdown(md)
    assert result.count("secret code") == 1


def test_prose_with_pipes_not_table():
    """2+ pipes in prose without table shape must be preserved."""
    text = "either a | b | c works"
    assert handle_markdown(text) == text


def test_headerless_table_rows_near_separator():
    """Rows without leading/trailing pipes count as table when adjacent to a
    separator row."""
    md = "Name | Age | City\n--- | --- | ---\nAlice | 30 | NYC"
    result = handle_markdown(md)
    assert "|" not in result
    assert "---" not in result
    assert "Alice" in result
    assert "NYC" in result


# ---------------------------------------------------------------------------
# CRLF handling (verifier findings: MULTILINE anchors assume \n)
# ---------------------------------------------------------------------------


def test_crlf_closed_atx_heading():
    """Closing-hash headings must strip on CRLF input, not leak '##'."""
    result = handle_markdown("## Heading ##\r\nbody")
    assert "#" not in result
    assert "Heading" in result
    assert "body" in result


def test_crlf_blank_line_stops_emphasis():
    """A CRLF blank line is a paragraph break — emphasis must not span it."""
    result = handle_markdown("*start\r\n\r\nend*")
    assert "*start" in result
    assert "end*" in result


def test_crlf_equivalent_to_lf():
    """CRLF input must normalize to the same speech text as LF input."""
    lf = handle_markdown("# Title\n\n**bold** and _em_\n\n- item")
    crlf = handle_markdown("# Title\r\n\r\n**bold** and _em_\r\n\r\n- item")
    assert crlf == lf


# ---------------------------------------------------------------------------
# Pathological-input performance (verifier finding: quadratic emphasis scan)
# ---------------------------------------------------------------------------


def test_unclosed_emphasis_flood_is_fast():
    """Repeated unclosed emphasis markers must not scan quadratically.

    100KB of '*word ' previously took >10s (O(n^2)); bounded emphasis
    spans make it linear. Generous ceiling to avoid CI flakiness.
    """
    import time

    for marker in ("*word ", "_word ", "__word "):
        payload = marker * (100_000 // len(marker))
        start = time.monotonic()
        handle_markdown(payload)
        assert time.monotonic() - start < 2.0


# ---------------------------------------------------------------------------
# Blockquoted fences and escaped pipes (verifier minors)
# ---------------------------------------------------------------------------


def test_fenced_code_inside_blockquote():
    """A fence opened inside a blockquote must not be garbled by the
    inline-code pass pairing backticks across lines."""
    result = handle_markdown("> ```\n> code line\n> ```")
    assert "`" not in result
    assert "code line" in result


def test_escaped_pipes_not_table():
    """Escaped pipes are literal content, not table syntax."""
    result = handle_markdown("use a \\| b \\| c here")
    assert result == "use a | b | c here"


def test_adversarial_floods_are_fast():
    """Every markdown pass must stay near-linear on pathological input.

    Covers the verifier-found quadratic vectors beyond emphasis: long
    dash lines (table separator backtracking), '[' floods (link/image
    scan-to-EOS), fence-opener and backtick-pair floods (per-placeholder
    str.replace restore loop).
    """
    import time

    payloads = [
        "-" * 40_000 + ".",          # table separator backtracking
        "[word " * (100_000 // 6),   # unclosed-bracket link flood
        "```\n" * (100_000 // 4),    # fence-opener flood -> placeholder restore
        "`a`" * (100_000 // 3),      # inline-code flood -> placeholder restore
    ]
    for payload in payloads:
        start = time.monotonic()
        handle_markdown(payload)
        assert time.monotonic() - start < 2.0, f"slow on {payload[:20]!r}..."


def test_emphasis_flood_scales_linearly():
    """Scaling ratio guard: 4x input must cost ~4x time (linear), not
    ~16x (quadratic). Ratio bound is generous for CI noise."""
    import time

    def cost(n: int) -> float:
        payload = "__word " * (n // 7)
        start = time.monotonic()
        handle_markdown(payload)
        return time.monotonic() - start

    small, large = cost(100_000), cost(400_000)
    assert large / max(small, 1e-3) < 9.0


def test_newline_and_whitespace_floods_are_fast():
    """List-marker patterns must not scan quadratically on blank-line
    floods ('^(\\s*)' + MULTILINE consumed all following newlines from
    every line anchor; 100KB of newlines took 65s)."""
    import time

    for payload in ["\n" * 100_000, "\r\n" * 50_000, " \n" * 50_000]:
        start = time.monotonic()
        handle_markdown(payload)
        assert time.monotonic() - start < 2.0


def test_long_wordrun_through_normalize_text_is_fast():
    """URL_PATTERN must not scan quadratically on long word-char runs
    with no dot-TLD (100KB single line previously hung normalize_text)."""
    import time

    from api.src.services.text_processing.normalizer import (
        normalize_text,
    )
    from api.src.structures.schemas import NormalizationOptions

    for payload in ["a" * 100_000, "a." * 50_000, "a@" * 50_000]:
        start = time.monotonic()
        normalize_text(payload, NormalizationOptions())
        assert time.monotonic() - start < 2.0
