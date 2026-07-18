# North Star: Correct, Open, Tunable Narration

*Distilled from the AI-singing/TTS landscape research (see `NARRATION_STRATEGY.md`).
This supersedes "singing" as the goal. Singing was the demo that taught us where
the real gap is; it stays a horizon toy, off the critical path.*

---

## The one sentence

**A pronunciation-and-prosody control layer that lets an author — or the LLM
drafting the script — guarantee *how* an open TTS says something, not just hope
it guesses right; correct by default, tunable when it isn't, and open enough to
give back.**

## What it is

- **Correct.** Jargon, acronyms, numbers, units, and names come out right the
  first time. Wrong pronunciation in training content isn't a polish issue — it's
  a correctness bug, and we treat it like one (tested, gated, regressible).
- **Tunable.** When the default is wrong, the author fixes it *once*, in a
  versioned lexicon, in the same phonetic language (IPA) the model already speaks
  through its G2P — not by misspelling words to trick the voice.
- **Open.** Built on Apache-2.0 Kokoro, self-hostable, no per-minute meter, data
  stays in-house. And the correctness layer itself is contributed back, not hoarded.

## What it is NOT

- **Not singing.** Vocoder-retune singing is physics-capped (V/UV boundary errors,
  formant–pitch coupling, degrades past ~150ms notes). Fun demo, permanent horizon.
- **Not a naturalness play.** We will lose a blind naturalness A/B to ElevenLabs
  and that's fine. Modern large TTS already infers good prosody from raw text; we
  don't out-scale them and won't try.
- **Not markup-for-its-own-sake.** Reserve explicit control for what scale
  provably *fails* at — pronunciation of jargon/acronyms/numbers and deliberate
  emphasis. Everything else, let the model do its job.

## Why this is the right long-horizon goal

The market wedge that survived adversarial scrutiny: **correctness + control +
cost, not naturalness.** Incumbents (ElevenLabs, Murf, WellSaid) patch
pronunciation *post-hoc, per word,* behind closed metered APIs. The dominant
authoring tool (Articulate Storyline) has **no pronunciation dictionary at all.**
No one lets the content-generation step emit the pronunciation/prosody markup as
part of drafting. That closed loop — LLM + phoneme-native open model + authoring
pipeline, owned end-to-end and open — is the defensible position, and it's most
valuable precisely for technical training content where a mispronounced acronym
is unacceptable.

## The good-faith open contribution

Kokoro today has no lexicon / pronunciation-control layer. The parts of our work
that *every* Kokoro user benefits from should go **upstream** (to Kokoro-FastAPI
or the misaki G2P project), not stay in our fork:

| Contribute upstream (good-faith) | Keep as our product |
|---|---|
| The Prosody-Annotated Phoneme IR (a clean token contract) | LMS/authoring integration (rippling-lms) |
| A pronunciation **lexicon resolver** + versioned lexicon format | Our domain lexicon (Oxford House jargon) |
| LLM→IPA **validation gate** (reject off-inventory phonemes) | The script-generation prompt chain |
| ReDoS/normalizer hardening *(already PR'd: remsky#489)* | The closed generation→narration loop |

The ReDoS fix we already sent upstream is the template: find a real gap in a
solid project, fix it cleanly and in isolation, give it back. The pronunciation
layer is the same move at larger scale — it makes Kokoro better for everyone and
makes our product possible at the same time.

## The shape of the contribution (concrete)

A **Prosody-Annotated Phoneme IR** as the contract between stages:

```
Token { text, phonemes|None, stress:-2..+2, emphasis:bool,
        break_after_ms:int, source: {g2p_default|lexicon|llm_authored|human_override} }
```

Pipeline (each stage swappable, upstreamable pieces marked *):
1. Structure/markdown normalizer  ← our md work, already built & hardened
2. Number/acronym/unit expander* (rule-based; where big TTS fails)
3. Lexicon resolver* (human > domain > rules > LLM > G2P default)
4. LLM IPA/prosody annotator + validation gate* (naive LLM→IPA is 31.6% PER;
   validated against Kokoro's phoneme inventory → ~5.8%)
5. Kokoro adapter (emit `[word](/ipa/)` + `[word](+N)`, chunk <40k, `speed` callable)

Graded by an **ASR-based eval harness** (WER against intended script) plus a
pronunciation unit-test corpus — build this first; it's what lets us *prove*
"correct," not just claim it.

## Hard-won constraints (don't relearn these)

- Kokoro stress markup `[word](+N)` works but is **subtle**; emotion markup
  **never** works (not in its training data) — don't build emotion on Kokoro.
- **Validate every LLM-authored IPA** against the phoneme inventory before
  injection; the failure mode is silent symbol inconsistency, not obvious garbage.
- Punctuation / respelling are the *reliable* prosody levers; SSML-style tags are
  widely ignored by modern open TTS.
- Phoneme injection breaks ~40k chars → chunk. Word-level timestamps are free
  forced alignment. `speed` takes a callable (unused pacing lever).
