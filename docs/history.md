# Releases

Models are named like the labs': a new name or number for a new recipe, a point release
(`.1`, `.2`) for a significant fix with the same recipe. `speedrun.sh` holds the id of the
next release, and the release gate compares it with the newest one in `models/`.

| model | recipe | what changed |
|---|---|---|
| `mini-1` | AdamW, a single RL run on every skill | the first release: stories and addition, 256-token context |
| `mini-2` | Muon, a math specialist distilled into the SFT model | story perplexity 7.37 → 6.39, 5-digit additions 91% → 100%, a new question after an answer 40% → 97% |
| `mini-2.1` | the specialist also practices multi-turn math | a follow-up on a 4-5 digit total 37% → 97% |
| `mini-3` | twice the pretraining (57M tokens), from a small [scaling law](training.md#lessons) | story perplexity 6.46 → 5.75, instructions 98% → 100% |
| `mini-3.1` | SFT and distillation also show something else after an answer | a story, a greeting or "Who are you?" after an addition: 28% → 99% |
| `mini-3.2` | requests after small talk in SFT; word problems and 5-digit totals for the specialist | a follow-up on a 5-digit total 63% → 91%; whole chats 85% |
| `mini-code-1` | a separate coding agent for opencode: pretraining on Python, SFT on agent transcripts | 98% of its coding tasks done end to end |
| `mini-4` | both in one model: Python in pretraining, transcripts and pretraining documents in SFT, a 1,024-token context | whole chats 85% → 92%, coding tasks 97.8%; plain stories 0.635 → 0.678 bits per character, shipped with a waiver |
| `prelude-1` | mini-4's recipe in a repo cut down to it: one chat template, the server fitting every prompt to the context | mini-4's results again (whole chats 90%, coding tasks 98%, stories 0.677 bits per character), and it says it is prelude; stories on topic 98% → 91%, shipped with a waiver |

Each release is a git tag on the commit that trained it, with its config and its docs as
they were. To reproduce one:

```bash
git checkout mini-3.2 && bash speedrun.sh small     # the presets of that commit: small, code, unified
```

What each change taught us is in [Lessons](training.md#lessons).
