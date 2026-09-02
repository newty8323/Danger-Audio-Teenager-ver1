# CLAUDE.md

Project: audio-harm-detection — audio-based classification of content harmful to minors (sexual / violence / gambling sounds) + risk scoring.

## Repo layout — two branches (since 2026-07-28)

- **`main`** (this branch) — only what the adopted 3-tier cascade needs to run, plus the
  Korean student-facing docs (`README.md`, `docs/01-overview.md` … `05-limits.md`).
- **`process`** — the full research history: rejected experiments, `spec.md`, `model_light.md`,
  `process.md`, `video_prompt.md`, lecture scripts.

Docs that live only on `process` are read with `git show process:<file>` (e.g.
`git show process:spec.md`). To WRITE to them, use a worktree — `git worktree add <tmp> process` —
never `git switch`, which would drop the untracked working files here.

**Full design = `git show process:spec.md`. Read it before any non-trivial task.**
Trap: `scripts/combined_data.py` and `scripts/train_beats_finetune.py` look like dead research
code but every adopted training script imports them. Never delete.

## Operating rules

### 1. Autonomy & verification
- Default: execute autonomously. Permission is pre-granted for routine work (code edits, tests, preprocessing runs, local training, analysis, doc updates).
- Before executing, self-verify the plan against `spec.md` (on `process`) using the strongest available reasoning (Opus 4.8 / extended thinking). If the plan conflicts with spec.md, stop and ask.
- ALWAYS ask the user first for **critical tasks**:
  - anything spending Kaggle GPU quota (`kaggle kernels push` with `enable_gpu`)
  - dataset collection/handling involving adult-content sources (ethics constraints, spec §6.4)
  - deleting data/checkpoints, force-push, changing class taxonomy or risk-policy weights
  - publishing/uploading anything public (dataset, code release)

### 2. Progress log — `process.md` (on the `process` branch)
- Append an entry after every meaningful unit of work (feature done, experiment launched/finished, decision made).
- Format: `## YYYY-MM-DD HH:MM — <title>` + 2–5 bullet summary (what/why/result/next). Never rewrite past entries.
- On session start: read the last ~3 entries — `git show process:process.md | tail -60`.
- To append: `git worktree add <tmp> process`, edit there, commit, push, `git worktree remove <tmp>`.

### 3. References — `references.md`
- Any paper/dataset/library doc actually used for a decision goes into `references.md`:
  `- [key] Author (Year). Title. Venue. — 1–2 line takeaway relevant to this project.`
- Deduplicate by key. Cite the key in code comments / process.md when a design choice comes from it.

### 4. Language
- All chat I/O with the user: **Korean**.
- Code, comments, commit messages: English. **Exception: the student-facing docs on `main`
  (`README.md`, `docs/0*.md`, `references.md` header) are Korean — they are read by students.**

### 5. Self-review (mandatory)
- After writing code or completing any task, review the work at least once with the
  strongest available reasoning (Opus 4.8 / extended thinking) before declaring it done.
- The review checks correctness against `spec.md` (on `process`), edge cases, and the conventions below.
- If the review finds issues, fix them and re-verify (re-run tests/lint). Only then report done.
- Prefer an independent pass (e.g. a fork/subagent review) over trusting the first draft.

## Conventions
- Python 3.11, PyTorch 2.x, hydra configs, ruff, pytest (preprocess & risk modules must have unit tests).
- Reproducibility: every experiment = hydra config + seed; artifacts (norm stats, thresholds, class weights) are versioned files, never hardcoded.
- Checkpoint/resume logic (`--resume auto`) is mandatory in every training script (Kaggle 12h sessions are the normal case, not an error).
- Commit style: `type(scope): summary` (feat/fix/exp/data/docs).

## Video production
- Any lecture-video work (subtitling, speed change, loudness, encoding of recordings in
  `Danger_audio_video/`): **ALWAYS read `video_prompt.md` first and follow it exactly**
  — `git show process:video_prompt.md` (STT model, silence-cut rule §9, subtitle text-cleanup rules,
  subtitle style, loudnorm, 1.2x, small-size encode, verification).
- When the user adds/changes a video requirement, append it to `video_prompt.md` on the `process`
  branch (keep history; never rewrite past rules).
- Video/subtitle files are never committed (gitignore covers folder + media extensions).

## Kaggle workflow (CLI)
- Data: precomputed log-mel `.npy` + manifest → Kaggle Dataset, versioned (`v1.0-base`, `v1.1-hnm1`, ...). Never upload raw adult-content audio.
- Run: `kaggle kernels push -p kaggle/` → poll `kernels status` → `kernels output` to retrieve ckpt/logs.
- Time guard inside training loop: save & exit at 11h elapsed.
- wandb key via Kaggle Secrets (read in-kernel), never committed.
- Quota (30h/week) is user-owned: report estimated GPU-hours before any push and get confirmation (see rule 1).
