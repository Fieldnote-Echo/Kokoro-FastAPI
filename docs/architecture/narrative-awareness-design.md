# Narrative Awareness — MVP Design

## Problem

Kokoro treats all input text as a flat stream of phonemes. Structural information — paragraph breaks, section dividers, headers, dialogue boundaries — is destroyed during normalization (`\n` → ` `, multiple spaces collapsed, `normalizer.py:563`). The resulting audio has no pacing variation between structural elements, making long-form narration (documents, articles, chapters) sound like a continuous run-on.

A skilled narrator reads *structure*, not just words. The audio should respect the topology of the text.

## Mental Model

A document has three structural layers:

1. **Macro** — chapters, major sections, title/headers
2. **Meso** — paragraphs, scene breaks, block quotes, lists
3. **Micro** — sentences, clauses, dialogue, parentheticals, emphasis

Kokoro currently operates at layer 3 only (the model handles within-sentence prosody via punctuation). Layers 1 and 2 are lost before the model ever sees the text.

Narrative awareness means producing a **narrator's score** — an annotated stream that preserves structural intent and translates it into audio behavior (pauses, chunk boundaries, and eventually voice parameter shifts).

## Architecture

### Annotation Vocabulary (MVP)

Five primitives, each with a default audio behavior:

| Primitive | Detection | Default Behavior |
|-----------|-----------|-----------------|
| `paragraph_break` | `\n\n` (double newline) | 0.6s silence |
| `section_break` | `---`, `***`, `___`, `\n\n\n+` | 1.2s silence |
| `heading` | Markdown `# ...` through `###### ...` | 1.0s silence before, 0.8s after, force chunk boundary |
| `dialogue` | Text enclosed in `"..."` or `\u201c...\u201d` (curly) | Force own chunk |
| `aside` | Text enclosed in `(...)` | Force own chunk |

Detection is pattern-based, deterministic, no LLM. Detection runs top-down: section breaks first, then paragraph breaks, then headings, then dialogue, then asides.

**Dialogue detection scope (MVP):** Double quotes only (straight `"` and curly `\u201c\u201d`). No single-quote dialogue (too ambiguous with apostrophes). No multi-paragraph dialogue handling. Dialogue detection runs before aside detection so parentheticals inside quotes don't split dialogue segments.

**Heading detection scope (MVP):** Markdown `#` syntax only. No all-caps detection, no HTML tags. Documented as a known limitation.

### Annotation Directionality

Annotations describe a **pause before** the segment's text. When `smart_split()` encounters a segment with annotations, it emits the pause *then* processes the text.

For headings, which need a pause on both sides, two approaches:
- The heading segment carries a `pause_before` annotation (1.0s)
- The heading segment also carries a `pause_after` annotation (0.8s), emitted after the heading text chunk

This is expressed by allowing multiple annotations per segment with a `position` field:

```python
@dataclass
class NarrativeAnnotation:
    """A structural annotation on a text segment."""
    kind: str                    # "paragraph_break", "section_break", "heading", "dialogue", "aside"
    pause_s: float               # silence duration in seconds (0.0 = no pause, just chunk boundary)
    position: str                # "before" or "after" — when to emit this pause relative to segment text
    force_chunk_boundary: bool   # if True, this segment must start a new chunk
    metadata: dict[str, Any]     # extensibility (heading level, quote depth, etc.)

@dataclass
class NarrativeSegment:
    """A segment of text with optional structural annotations."""
    text: str                           # the actual text content
    annotations: list[NarrativeAnnotation]  # structural annotations (can be empty)
```

### Pause Collision Policy

When multiple pause sources coincide, use **max() semantics** — take the longest pause, don't stack.

| Scenario | Resolution |
|----------|-----------|
| Section break (1.2s) + heading before (1.0s) | max(1.2, 1.0) = 1.2s |
| Paragraph break (0.6s) + explicit `[pause:2s]` | Explicit tag wins: 2.0s |
| Heading at document start | Skip the "before" pause (no pause before first segment) |
| Two annotations both `position="before"` | max() of their pause_s values |

Explicit `[pause:Xs]` tags in source text **override** (not stack with) structural pauses at the same boundary. Rationale: if the author specified an explicit pause, they know what they want.

### Paragraph Segmentation

When dialogue or aside is detected within a paragraph, the surrounding text becomes its own segment(s):

```
Input:  He said "hello" and left.

Output:
  NarrativeSegment(text='He said', annotations=[])
  NarrativeSegment(text='"hello"', annotations=[
      NarrativeAnnotation(kind="dialogue", pause_s=0.0, position="before", force_chunk_boundary=True, metadata={})
  ])
  NarrativeSegment(text='and left.', annotations=[])
```

For a paragraph that IS dialogue: `"Yes," he said.` — the entire paragraph is one dialogue segment (the quotes span the full text).

For the novel's italic-rendered speech (*it's still falling*), no special detection in MVP. Markdown italics are stripped during normalization but don't trigger dialogue chunking.

### Pipeline Integration — String Injection Approach

After investigating `smart_split()`, the **safest MVP path is string injection** rather than modifying `smart_split()`'s signature.

**Rationale:** `smart_split()` (text_processor.py:136-325) is the core chunking engine. It already handles `[pause:Xs]` tags by splitting on `PAUSE_TAG_PATTERN` first (line 155), then processing each text part through normalization and sentence splitting. The pause infrastructure works. Changing its signature from `str` to `list[NarrativeSegment]` touches every caller and every test, with high regression risk for a behavioral change that can be achieved without it.

**MVP approach:**
1. `annotate()` takes raw text, detects structure, produces `list[NarrativeSegment]` internally
2. `segments_to_tagged_text()` converts the segment list back to a string with `[pause:Xs]` tags and chunk boundary markers injected at the right positions
3. The tagged string is passed to `smart_split()` unchanged

This keeps `NarrativeSegment` as the annotator's internal representation (good for testing and future extensibility) while outputting in the vocabulary the pipeline already speaks.

**Chunk boundary forcing** is achieved by injecting a `[pause:0.001s]` tag, which forces `smart_split()` to yield the current chunk before continuing (the regex split creates a new text part). This is a minor hack but works with the existing machinery.

**Future:** When the system needs voice parameter shifts or rate changes, `smart_split()` gets upgraded to accept `list[NarrativeSegment]` directly. The annotator's internal representation doesn't change — only the output adapter.

#### Pipeline flow:

```
raw text → annotate() → NarrativeSegment[] → segments_to_tagged_text() → str with [pause:Xs] tags → smart_split() → ...existing pipeline...
```

#### Entry point:

`annotate()` is called inside `tts_service.py:generate_audio_stream()` on the raw text, before passing to `smart_split()`. This is the single entry point for all audio generation (line 289). The development router imports `smart_split` but doesn't call it directly with user text.

### Default Pause Durations

Configurable via settings (environment variables), following the existing pattern in `config.py`:

```python
# Narrative awareness defaults
narrative_paragraph_pause: float = 0.6     # seconds between paragraphs
narrative_section_pause: float = 1.2       # seconds at section breaks
narrative_heading_before_pause: float = 1.0  # seconds before heading
narrative_heading_after_pause: float = 0.8   # seconds after heading
```

Dialogue and aside don't get configurable pause — they only force chunk boundaries.

### Detection Order

Within `annotate()`, detection runs in this order:

1. **Section breaks** — split on `---`, `***`, `___`, `\n\n\n+`
2. **Paragraph breaks** — split on `\n\n` within each section
3. **Headings** — detect lines starting with `#` (must be first non-empty line in a paragraph segment)
4. **Dialogue** — detect `"..."` / `\u201c...\u201d` spans within paragraph text
5. **Asides** — detect `(...)` spans within non-dialogue text

Each layer refines segments produced by the previous layer. Dialogue runs before asides to prevent parentheticals inside quotes from splitting dialogue.

## Files Changed

| File | Change |
|------|--------|
| `api/src/services/text_processing/narrative_annotator.py` | **New.** `annotate()`, `segments_to_tagged_text()`, dataclasses |
| `api/src/services/tts_service.py` | Call `annotate()` + `segments_to_tagged_text()` before `smart_split()` |
| `api/src/core/config.py` | Add narrative pause duration settings |
| `api/src/services/text_processing/text_processor.py` | **No change** |
| `api/src/services/text_processing/normalizer.py` | **No change** |
| Tests | New test file for annotator |

## Test Strategy

1. **Annotator unit tests** — feed known markdown structures, verify `NarrativeSegment` list output. Pure input/output, no mocks needed. Test cases from the novel:
   - `"Cold out there," she said.` — dialogue detection
   - `He swept (as always) and oiled the bench.` — aside detection
   - `# Chapter One\n\nParagraph.` — heading + paragraph break
   - `---` between paragraphs — section break
   - Curly quotes from Substack HTML-to-text
2. **Tagged text output tests** — verify `segments_to_tagged_text()` produces correct `[pause:Xs]` strings with collision resolution
3. **Round-trip regression tests** — existing test inputs produce identical output (plain text without structure should be unchanged)
4. **Structural audio duration tests** — document with structure produces longer audio than same text with structure stripped (proves pauses are being inserted)
5. **Edge cases** — empty input, single paragraph, all headings no body, text with existing `[pause:Xs]` tags, document starting with heading

## What MVP Does NOT Include

- No LLM-based structure detection (pattern matching only)
- No voice parameter shifts per segment (just pauses and chunk boundaries)
- No rate/speed changes per segment
- No emphasis or italic detection
- No single-quote dialogue detection (apostrophe ambiguity)
- No multi-paragraph dialogue handling
- No heading detection beyond markdown `#` syntax
- No nested annotation resolution (flat list is sufficient for MVP)
- No SSML support
- No per-request configuration of pause durations (deployment-level only)

## Success Criteria

1. Narrating a markdown document with headers, paragraphs, and section breaks produces audible pacing differences between structural elements
2. Narrating prose with dialogue keeps quoted speech in coherent chunks
3. Existing `[pause:Xs]` tags continue to work unchanged
4. No regression in short-form / non-structured text (plain sentences still sound the same)
5. Pause durations are configurable via environment variables without code changes
6. `smart_split()` signature and behavior is completely untouched
