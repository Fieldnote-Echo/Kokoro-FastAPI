# Fork Roadmap

*Why this fork exists, and where it's going. Full rationale in [`NORTH_STAR.md`](./NORTH_STAR.md); market/technical research in [`NARRATION_STRATEGY.md`](./NARRATION_STRATEGY.md).*

## North star

**Correct, open, tunable narration.** Kokoro can *say* words; it can't yet
*guarantee* it says them correctly, and no open TTS gives authors real control
over pronunciation. This fork's purpose is to close that gap — pronunciation and
prosody an author (or the LLM drafting the script) can guarantee, correct by
default and tunable when it isn't — and to contribute the generally-useful parts
back upstream.

**Not** a naturalness contest (large TTS wins that; fine) and **not** singing
(a physics-capped demo, permanently off the critical path). The wedge is
**correctness + control + cost.**

## Good-faith upstreaming

Parts that help every Kokoro user go to `remsky/Kokoro-FastAPI`; the product
(LMS integration, domain lexicons, the generation→narration loop) stays here.
The ReDoS fix ([remsky#489](https://github.com/remsky/Kokoro-FastAPI/pull/489))
is the template: find a real gap, fix it cleanly in isolation, give it back.

## Status

| # | Milestone | State |
|---|---|---|
| 0 | Upstream sync to v0.6.0 | ✅ done |
| 0 | Supply-chain hardening (digest pins, SHA-pinned actions, provenance, SBOM, Trivy) | ✅ done |
| 0 | Normalizer ReDoS hardening | ✅ done · upstream PR #489 |
| 1 | Markdown-aware normalization (LLM output → speakable) | ✅ done |
| 2 | **ASR-based eval harness + pronunciation test corpus** | ⏭️ next |
| 3 | Deterministic number/acronym/unit expander | planned |
| 4 | Versioned pronunciation **lexicon** + resolver | planned |
| 5 | Prosody-Annotated Phoneme **IR** (the token contract) | planned |
| 6 | LLM IPA/prosody annotator **behind a validation gate** | planned |
| 7 | Blind A/B vs raw Kokoro + one commercial baseline | planned |
| H | *(horizon)* speech-to-singing WORLD retune — demo only | spike done |

## Milestone 2 first (why)

You can't claim "correct" or "less robotic" without measuring it. The harness
(WER via ASR against the intended script + a jargon/acronym unit-test corpus)
grades everything downstream and instantly catches pronunciation regressions.
It's the smallest, highest-leverage piece and it unblocks every later A/B claim.

## Constraints banked (don't relearn)

- Kokoro stress markup `[word](+N)` works but is **subtle**; emotion markup
  **never** works (not in its training data).
- **Validate LLM-authored IPA** against Kokoro's phoneme inventory before
  injection — naive LLM→IPA is ~31.6% phoneme-error; validated ~5.8%.
- Punctuation/respelling are the reliable prosody levers; SSML-style tags are
  widely ignored. Phoneme injection breaks ~40k chars → chunk. Word timestamps
  are free forced alignment. `speed` takes a callable (unused pacing lever).
