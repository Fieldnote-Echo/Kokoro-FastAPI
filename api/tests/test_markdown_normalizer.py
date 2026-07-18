"""Unit tests for handle_markdown() — the navi-os fork of Kokoro's normalizer."""

from api.src.services.text_processing.normalizer import handle_markdown

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
