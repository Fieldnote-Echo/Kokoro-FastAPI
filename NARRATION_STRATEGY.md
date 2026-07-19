# Decision Brief: Kokoro-Based Natural Narration (and the Singing Horizon)

*Synthesized from 5 research angles + adversarial verification. Where a verifier corrected a finding, the corrected version is what appears below — several load-bearing claims changed under scrutiny, flagged inline as **[verified]** or **[corrected]**.**

---

## 1. HOW AI SINGING IS DONE — Taxonomy & Where We Sit

There are three distinct families. They differ primarily by **what you feed in at inference time**.

| Family | Input at inference | Training data needed | Quality ceiling | Examples |
|---|---|---|---|---|
| **SVS (score-trained)** | Symbolic music score (phoneme + pitch-ID + note-duration streams) + lyrics. **No reference audio.** | Paired score/singing corpus per voice | Highest for singing; still fighting for every 0.5 MOS | DiffSinger, VISinger2, NNSVS; commercial: Synthesizer V, ACE Studio |
| **SVC (voice conversion)** | An **existing sung performance (audio)** + target-speaker embedding. Never sees a score. | ~10 min target-voice audio | High; preserves source expressivity | RVC (MIT), so-vits-svc (AGPL-3.0) |
| **Speech-to-singing / vocoder-retune (OURS)** | A **spoken utterance** + a target melody (note pitches/durations). DSP F0 substitution. | **Zero.** No neural training at all. | Lowest of the three — structural ceiling | WORLD/STRAIGHT pipelines; Saitou & Goto 2007 |

**Key architectural facts (verified):**
- SVS systems (DiffSinger, VISinger2, NNSVS) genuinely take a symbolic score and need **no reference audio at inference** — this is the clean distinction from SVC. **[corrected]** Caveat: their singer *timbre* is still learned from training-time audio, and a 2025-26 zero-shot SVS branch (e.g. "Everyone-Can-Sing") *reintroduces* an inference-time reference clip for timbre cloning. So "no reference audio" is a property of *classic* score-driven SVS, not the category's future.
- DiffSinger's shallow-diffusion trick (start reverse diffusion partway using a cheap aux decoder's mel prediction, not pure noise) is **[verified]** exactly as described; openvpi fork is Apache-2.0, 44.1kHz, swappable vocoders.
- **[corrected]** NNSVS's autoregressive F0 model is framed by its own paper as an *alternative to* explicit vibrato modeling ("can generate a more natural voice without explicitly modeling vibrato"), **not** a dedicated vibrato mechanism. Minor, but don't cite it as "the vibrato model."
- RVC/so-vits-svc pipeline (content encoder → F0 extractor (RMVPE) → VITS → NSF-HiFiGAN) is **[verified]**. License gap is real and matters: **RVC is MIT, so-vits-svc is AGPL-3.0** (network-use source-disclosure obligation).

### Where OUR speech-to-singing pipeline sits — and its honest ceiling

Our recipe (TTS word-timestamps as forced alignment → WORLD F0-flatten-to-notes + vibrato → frame-stretch to note durations, zero training) is a **modern instance of the Saitou & Goto 2007 STRAIGHT approach**, now on WORLD. That lineage is real and works.

**The ceiling is structural, not a tuning problem [verified]:**
- **V/UV boundary errors:** WORLD estimates F0 (DIO), spectral envelope (CheapTrick), aperiodicity (D4C) as *independent* streams. They can disagree about which frames are voiced at exactly the note onsets/offsets a singing pipeline cares about → audible discontinuities.
- **Formant-vs-pitch coupling:** harmonics are physically tied to F0. Forcing a new pitch onto an envelope estimated for the *old* pitch introduces the "buzzy"/distorted artifact — a 2026 paper (arXiv:2601.10345) exists purely to *repair* pitch-shift-degraded singing, i.e. this is treated as an unsolved problem.
- **Regime limits:** STS quality degrades on notes >150ms and beyond the speaker's corpus vocal range — exactly the sustained-note, wide-interval regime narration-to-singing hits.
- **Trained SVS itself tops out ~2.85–4.0 MOS** (ground-truth-vocoded ceiling ~4.04). A retune-of-speech approach sits *below even a naive trained SVS baseline*. Expect meaningfully lower, especially on sustained notes and vibrato.

**[corrected] One myth to drop:** the "singing F0 range ~80–3400 Hz" figure is wrong — it conflates fundamental frequency (realistically ~65–1500 Hz) with the **singing formant** (a ~2–3.4 kHz spectral resonance). Related **[corrected]** nuance: DSP retuning *can* engineer singing-specific timbre cues — Saitou 2007 explicitly hand-builds a singer's-formant boost and vibrato-synced formant AM, and rated near-real-singing in listening tests. So the honest framing is not "DSP *cannot* model singing timbre" but "DSP models it via **hand-engineered rules**, not **learned** from data — which caps flexibility on subtler cues (breathiness, register transitions, singer idiosyncrasy)."

**Bottom line:** Our approach's *only* durable advantage is **zero singing-training-data + any TTS voice retunable**. It is the right tool for a demo/novelty/horizon feature, the wrong tool to compete on singing quality. Treat it exactly as the concept already does: **horizon, not deliverable.**

---

## 2. OUR CONCEPT: BETTER / WORSE / HOPEFUL

### Tier 1 — md→phoneme/prosody usability layer (the NEAR-TERM GOAL)

**This is the right bet, but it is low-novelty — that's fine, novelty isn't the goal, natural LMS narration is.**

- **WORSE / already solved by others:** The pattern of "have the upstream LLM emit inline IPA + emphasis markup before text hits the TTS" is **already shipped vendor practice** — Inworld AI's docs literally instruct developers to prompt their LLM to wrap emphasis in asterisks and replace proper nouns with IPA. ElevenLabs v3 ships bracketed Audio Tags for the same seam. LLM-generated SSML tag placement for narration is a *published, quantified* result (ICNLSP 2025: ~50% listener-preference lift on French audiobooks with Qwen). So we are not inventing a category.
- **BETTER (our actual edge):** every one of those is **closed** and solves pronunciation as a **post-hoc, per-word patch** (pronunciation dictionaries, SSML `<phoneme>`, phonetic respelling) **disconnected from the content-generation step**. Articulate Storyline 360 — the dominant authoring tool — *has no pronunciation dictionary at all*, forcing authors to misspell words phonetically. The whitespace: **the LLM that drafts the course script emits the phoneme/prosody markup natively, as part of drafting**, into a phoneme-native open model we control end-to-end. Nobody ships that closed loop.
- **HOPEFUL but honest — scale is partly eating this:** Amazon's BASE TTS shows that at 10K+ training hours, "emergent" prosody appears and models infer correct *contextual/segmental* prosody from raw text with **no markup**. So the "not robotic" problem is *partly* solved by just using a good model. BUT the same paper shows scale does **not** solve emotional/emphasis intent (stuck ~2.0/3 even at max scale) or paralinguistics without explicit keywords. **Implication:** for plain "read this clearly and naturally," lean on the model + light normalization; reserve explicit markup for the things scale demonstrably *doesn't* fix — jargon pronunciation, acronyms, numbers, deliberate emphasis. Don't over-engineer prosody markup where a good voice already handles it.

### Tier 2 — phoneme-native LLM output (the BET / HORIZON)

**Mechanism is real and researched; the specific "external LLM hand-authors a score a frozen TTS honors" variant is unproven — and Kokoro specifically is a weak substrate for it.**

- **BETTER / real precedent:** The decomposition "LM predicts explicit prosody tokens *first*, then conditions speech generation" is proven to help (RALL-E: WER 6.3%→2.8%; ProsodyLM; BatonVoice's "conductor LLM emits textual vocal-feature plan, BatonTTS performs it" is architecturally *our exact shape*). DiffSinger's phoneme+pitch-ID+note-duration input **is** the "explicit prosody score" interface Tier-2 wants — proving our concept rediscovers an established pattern rather than inventing one.
- **WORSE / the falsifying reality:** In *every* published case the prosody tokens are predicted by **the TTS system's own internal LM**, not authored by an upstream general-purpose LLM and honored by a frozen TTS. **Nobody has shown external-authored prosody tokens honored by a frozen model.** And the field's momentum is the *opposite direction* — natural-language style prompts (Parler-TTS, PromptTTS2, InstructTTS), i.e. describe prosody in English prose, not phonetic markup.
- **Kokoro-specific constraints (heavily revised under verification):**
  - Phoneme injection `[word](/ipa/)` **works [verified]** — but **breaks on very large inputs** (~40k chars: model speaks *both* the word and the phonetic hint). **Mitigation: chunk text.** This is a real operational constraint, not a blocker.
  - **[corrected — important]** The earlier "stress markup is non-functional" finding is **wrong**. Issue #170 is *closed*; the maintainer confirmed stress changes work on ≥0.9.4, and a second user confirmed the effect **does occur but is subtle and sentence-dependent** ("don't expect miracles"). So Kokoro *does* respond to `[word](+1/+2/-1/-2)` stress — just weakly. This is a *usable* signal for light emphasis, not a dead end.
  - **[corrected]** The "SSML request unanswered" finding is **refuted** — issue #36 got a maintainer reply (2025-02-04) pointing to markdown pronunciation syntax + Lexicon overrides as the sanctioned surfaces, and noting emotion is blocked by **training data** (never trained on sad/angry), not by missing SSML parsing. Takeaway: emotion control won't come from markup on Kokoro at all.
  - **[verified]** `speed` is `Union[float, Callable[[int],float]]` — a callable gives **per-segment** rate control (not just one global float). Under-appreciated lever.
  - **[verified]** Kokoro inherits StyleTTS2's internal per-phoneme ProsodyPredictor (duration/F0/energy) but it is **not exposed** as a settable input. FastSpeech2 is the cleanest open model that *does* expose inference-time pitch/energy/duration control — but only as **scalar multipliers**, not arbitrary contours. If we ever want true per-phoneme control, forking Kokoro to expose the ProsodyPredictor (FastSpeech2-style) is the architectural template.

**Tier-2 verdict:** Genuinely interesting, genuinely unproven, and Kokoro is not the model that will validate it (StyleTTS2's learned diffusion prosody prior may *fight* injected scores rather than compose with them — an open empirical question). Keep Tier-2 as a **research spike behind the Tier-1 product**, not a roadmap commitment. What reliably works as markup across modern open TTS is **discrete trained-in tags** ([laugh], [S1]) — Orpheus/CSM/Dia — not continuous prosody values. That tells us the honorable path is discrete, trained-in vocabulary, not free-form pitch curves bolted onto a frozen model.

---

## 3. MARKET NICHE — Sharpest Defensible Position

**The niche: "Correctness-first, self-hostable course narration where the script generator and the voice speak the same phonetic language."**

Four combined properties none of the incumbents hold together:
1. **Open-source + Apache-2.0** (Kokoro) — self-hostable, no per-minute metering, data stays in-house.
2. **Phoneme-controllable** with an explicit IPA injection port.
3. **LLM-integrated at generation time** (markup authored *during* drafting, not patched after).
4. **LMS-embedded** — sits inside the course-authoring pipeline, not a separate voiceover SaaS.

**Who we beat, and on what axis:**

| Competitor | We beat them on | Their advantage over us |
|---|---|---|
| **ElevenLabs / Murf / WellSaid / PlayHT** | Cost at scale (they meter $0.15–0.30/finished-min; we self-host at ~zero marginal), data residency, generation-time pronunciation vs. their post-hoc per-word patching | Raw naturalness, voice cloning, polish |
| **Synthesia** | Price (~$2.90–3.00/min — video-bottlenecked), pure-audio focus | Talking-head video (different lane) |
| **Articulate Storyline 360 / Adobe Captivate (Polly/Azure behind them)** | They have *no* (Storyline) or legacy (Captivate) pronunciation dictionary; we make jargon/acronym correctness a first-class generation feature | Incumbent LMS integration, install base |
| **Piper / Coqui XTTS (open alternatives)** | Piper pivoted to **GPL-3.0** (Oct 2025) and has no phoneme injection; Coqui's good model (XTTS-v2) is **non-commercial** weights and dropped espeak-ng | Coqui has voice cloning |

**The defensible axis is CORRECTNESS + CONTROL + COST, not naturalness.** We will lose a blind naturalness A/B to ElevenLabs. We win on: *"technical training content where mispronounced jargon/acronyms/numbers is unacceptable, generated at LMS scale without per-minute fees, self-hosted."* Kokoro's training distribution is **long-form reading/narration [verified]** — already the right style for course VO, a poor fit for conversational competitors. Market is large and growing (AI-voice ~$5.6–7.7B in 2026 → $21–33B by 2030-32; every vendor ships a pronunciation-editor UI, proving demand) but **no one lets the content-generation step emit the markup**. That is the wedge.

**Honest caveat:** "correctness-first" is a real but *narrow* wedge — it's a feature, not yet a moat. The moat is the closed loop (LLM + phoneme model + LMS) being *ours end-to-end* and *open*.

---

## 4. BUILD PRIMITIVES & LESSONS — Actionable Modules

### 4.1 Core data structure: the Prosody-Annotated Phoneme IR

A single intermediate representation everything else operates on. Proposed shape — a list of tokens:

```
Token {
  text: str            # original grapheme span
  phonemes: str|None    # IPA override (None = let misaki G2P handle it)
  stress: int           # -2..+2, maps to Kokoro [word](+N) — WORKS but subtle
  emphasis: bool        # drives punctuation/respelling tactics, not a fake tag
  break_after_ms: int   # realized via punctuation/ellipsis, not SSML
  source: enum          # {g2p_default, lexicon, llm_authored, human_override}
}
```
`source` provenance is load-bearing — it lets the eval harness attribute regressions and lets human overrides always win. This IR is the contract between the LLM normalizer, the lexicon, and the Kokoro adapter.

### 4.2 Pipeline stages (each a swappable module)

1. **Markdown/structure normalizer** — strip/interpret markdown, expand lists/headings into speakable prose, handle code blocks (spell or skip).
2. **Number/acronym/unit expander** — deterministic, rule-based FIRST (dates, currency, units, "API"→"A P I" vs "NASA"→"nassa"). This is where scale-based TTS *fails* (BASE TTS: acronyms read letter-by-letter), so it's high-value and cheap.
3. **Lexicon resolver** — see 4.4.
4. **LLM prosody/IPA annotator** — fills `phonemes`/`stress`/`emphasis` for spans not covered by lexicon. Must be prompt-engineered + post-validated (see lesson below).
5. **Kokoro adapter** — emits `[word](/ipa/)` + `[word](+N)` + punctuation, **chunks to <~40k chars**, uses the `speed` callable for per-segment pacing.
6. **(Horizon) WORLD retune module** — separate, off the critical path.

### 4.3 Eval harness — how to A/B naturalness (build this EARLY)

- **Objective gate first:** run TTS output back through **ASR (WER)** against the intended script. Catches the failures that actually matter for e-learning — wrong pronunciations, dropped words, the injection-doubling bug. Cheap, automatable, CI-able.
- **Pronunciation unit tests:** a fixed corpus of jargon/acronym/number cases with expected phoneme or ASR-transcription targets. Regression suite for the lexicon + expander.
- **Human A/B (MOS/preference):** pairwise (our-normalized vs. raw-Kokoro vs. ElevenLabs) on real course paragraphs. Small n, forced-choice preference is more sensitive than absolute MOS.
- **[verified] Do NOT reuse speech-MOS models for any singing output** — SingMOS-Pro shows speech-MOS models fail on singing (domain gap). Singing needs its own raters.
- Instrument by IR `source` so you can say *which stage* moved the needle.

### 4.4 Pronunciation lexicon for jargon/acronyms

- **Format:** term → IPA (+ optional part-of-speech/context key for homographs). Store as versioned data, not code.
- **Resolution order:** human override > domain lexicon > deterministic acronym/number rules > LLM annotation > misaki G2P default.
- **[corrected] Validate LLM-authored IPA before injection.** LLM-G2P benchmark: naive prompting = **31.6% phoneme error rate**; optimized prompting + dictionary post-processing brought Claude 3.5 Sonnet to **5.8% PER**. Dominant failure is **inconsistent symbol choice for the same sound** (silent corruption), not obvious hallucination. So: validate LLM IPA against **Kokoro/misaki's actual phoneme inventory** and reject/normalize off-vocabulary symbols. Compounding-error risk: espeak-ng (Kokoro's fallback) itself mishandles homographs and foreign words, and **there's no downstream repair once bad phonemes enter the TTS.**
- **[corrected] Consider "LLM improves the G2P, not LLM in the inference loop."** "Fast, Not Fancy" (2025) argues augmenting fast rule-based G2P with LLM-*generated training data* beats calling an LLM at inference. Cheaper, lower-latency, deterministic at runtime — a strong option for the lexicon-building phase.

### 4.5 Lessons already learned (bank these)

- **[corrected] Kokoro responds to stress markup — but subtly.** `[word](+1..+2/-1..-2)` works (issue #170 closed, maintainer + user confirmed) on short/less-stressed words, effect is subtle and sentence-dependent. Usable for light emphasis; don't expect strong control.
- **[corrected] Kokoro does NOT do emotion via markup — ever.** Blocked by training data (never trained on emotion categories), per maintainer. Don't build emotion control on Kokoro; it needs a different model or fine-tune.
- **[verified] Phoneme injection is reliable at normal input sizes, breaks ~40k chars → chunk.**
- **[verified] `speed` accepts a callable for per-segment pacing** — free pacing control, currently unused.
- **Punctuation, ellipses, and phonetic respelling are the *reliable* prosody levers** across neural TTS; SSML-style tags are widely ignored. Prefer orthographic tactics over markup where possible.
- **[verified] WORLD retune works but has a hard, structural ceiling** (V/UV boundary errors, formant-pitch coupling, degrades >150ms notes / beyond corpus range). Horizon feature only. And drop the bogus "80–3400Hz F0" figure.
- **Forced alignment is free** — Kokoro's word-level timestamps *are* the alignment; no aligner needed for the retune pipeline.
- **Discrete trained-in tags are the only markup that reliably works** in modern open TTS (Orpheus/CSM/Dia). If we ever want robust expressive control, the path is trained-in vocabulary, not free-form continuous markup on a frozen model.

---

## 5. RECOMMENDED FIRST MOVES (ordered)

1. **Build the ASR-based eval harness + pronunciation unit-test corpus first.** You cannot claim "less robotic" or "correct" without measurement, and this catches the injection-doubling and jargon failures immediately. Everything downstream is graded against it. *(Smallest, highest-leverage, unblocks all A/B claims.)*

2. **Ship the deterministic normalizer + jargon/acronym/number lexicon** (pipeline stages 1–3, IR from 4.1). This is where scale-based TTS provably fails and where our correctness wedge lives — rule-based, no LLM/model risk, immediately demoable on real Oxford House course scripts.

3. **Add the LLM IPA/prosody annotator with a validation gate** (stage 4 + 4.4). Prompt-engineer + post-validate against Kokoro's phoneme inventory; measure PER against the corpus from step 1. Gate: must beat raw-Kokoro on the eval before it ships. Keep it *narrow* — jargon/emphasis where scale doesn't help — rather than annotating everything.

4. **Run the first blind A/B** (our-normalized vs. raw-Kokoro vs. one commercial baseline) on real course paragraphs. Decide with data whether the Tier-1 layer actually moves naturalness/correctness enough to build the product around, and *which stage* did the work.

5. **Time-box a Tier-2 spike (research, not product):** test whether Kokoro *honors* externally-authored stress/prosody markup at all beyond the known-subtle stress effect — one afternoon, tight hypothesis, using the harness. Result feeds the "is the phoneme-native bet real on this model?" question without committing roadmap. Keep the WORLD singing pipeline entirely off the critical path as the demo/horizon artifact.

**Guiding principle:** the near-term goal (natural, *correct* LMS narration) is achievable, defensible, and mostly *engineering*, not research. The horizon (phoneme-native LLM output, singing) is genuinely interesting but unproven and — for singing especially — capped by physics on our chosen approach. Fund the goal; spike the horizon; don't confuse them.