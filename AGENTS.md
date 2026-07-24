# mybot — AGENTS.md

> Minimal staged copy of [nananobot](https://github.com/anomalyco/opencode)'s agent subsystem, built for **learning by reproduction**.

## Working principles (NON-NEGOTIABLE)

These six rules outrank any other consideration. If a request seems to
conflict with them, surface the conflict before coding.

### 1. Minimal increment per stage
- One stage = **one** small, coherent capability.
- Prefer 20–50 lines over 200 lines. If a stage needs more, split it.
- No premature abstraction. Build the concrete thing first; extract only
  when a second concrete user appears.
- No opportunistic refactoring inside a feature stage. Save it for an
  explicit refactor stage.

### 2. Complexity cap per stage
- Resist "while we're here, let's also add X". Defer X to its own stage.
- Avoid frameworks / SDKs / patterns that pull in >100 lines of incidental
  complexity. A 1700-line upstream provider is a **learning target**, not
  a thing to copy wholesale at this point.
- When a feature has 3 sensible shapes, pick the dumbest one and move on.

### 3. Final goal = align with nananobot
Every stage should leave the project **structurally closer** to
nananobot's agent subsystem, not laterally divergent. Before adding a
new module, ask:

> *"Where does this live in nananobot, and does my placement match?"*

If mybot invents a new abstraction nananobot doesn't have, that's a
**red flag** — either we're wrong, or we should record why we're
intentionally diverging in `DIVERGENCES.md`.

### 4. Commit & push at every stage boundary
A stage is **not finished** until it's landed on `origin/main`. After
the green test, immediately:

```
git add <touched files only>
git commit -m "Stage N: <one-line summary>"
git push origin main
```

Rules:

- One commit per stage. Don't bundle unrelated changes.
- The commit message must start with `Stage N:` so the log stays
  scannable as a stage sequence.
- `.env`, `__pycache__/`, `.egg-info/` etc. must never enter a commit.
  Verify with `git status` before every commit.
- "It works on my machine" is not done. Pushed == done.
- A stage with uncommitted work is a stage that doesn't exist yet.

### 5. Explain approach before implementing
For any non-trivial stage (≥ ~20 lines of new logic, or new
abstractions), the agent MUST lay out the approach **in prose**
before writing code:

- What problem we're solving.
- What alternatives were considered.
- Which we picked and why.
- The minimum code shape (function signatures, key data structures).
- How we'll verify it.

Only after the user confirms (or adjusts) do we switch to writing
files. This rule does NOT apply to one-line fixes, trivial edits,
or pure exploration/reading tasks.

### 6. Feature must be locally runnable before it's "done"
A stage is not done until the new capability actually works end-to-end
on this machine. After writing the code, the agent must:

- For pure logic (parsers, formatters, hooks): a smoke test or
  `python -c "..."` invocation that exercises the new code path.
- For network-touching features: at least one real-API run that
  demonstrates the behavior, with the result captured in the commit
  message or a follow-up note.
- For UI / REPL features: a piped-input run that exits cleanly and
  shows the expected turn summary.

"Tried it once and it seemed to work" is not enough — the test must
be reproducible from the project's installed state. If a feature
requires a key or external service, the test command should make the
absent-key failure mode explicit (e.g., print a clear "MYBOT_API_KEY
is not set" message) so a CI / fresh-clone smoke run can detect
regressions.

## Project context

- Python 3.11+, asyncio, stdlib `logging` only (no loguru).
- Layout: `src/mybot/{agent,providers,tools,session}/`, `examples/` for demos.
- Real deps: `httpx`, `python-dotenv`. SDKs (openai, anthropic, …) are
  intentionally **not** installed — the OpenAI-compat path talks raw
  HTTP for now.
- Persistence: `~/.mybot/sessions/<key>.jsonl`, created on first save.

## Divergences from nananobot (intentional, for now)

| Area | mybot today | nananobot | When to revisit |
|---|---|---|---|
| OpenAI SDK | `httpx.AsyncClient` raw | `openai.AsyncOpenAI` + lazy import | After we've learned the wire protocol |
| Provider count | 2 (mock + openai_compat) | 12+ | As needed per real model usage |
| Hooks system | None | `HookManager` with pre/post LLM, pre/post tool | Stage ~8 |
| Session / memory | JSONL persistence only; no metadata, no cursor, no TTL | `SessionManager` with metadata, `last_consolidated`, idle-TTL | 7.1+ (metadata), 7.2+ (cursor), 8+ (TTL) |
| Message compaction | count-based truncation + LLM summary on REPL run | TTL-based auto-compaction in `Manager.compact()` | nananobot's TTL trigger can be added once a timer source is wired in (post-Hooks) |

## Stage log

| # | Topic | Status |
|---|---|---|
| 0 | Minimal LLM call | ✅ |
| 1 | Streaming | ✅ |
| 2 | Tool execution loop | ✅ |
| 2.5 | Provider + Tool abstractions | ✅ |
| 3 | Mock providers + sample tools + runnable demo | ✅ |
| 4 | OpenAI-compatible real-API provider | ✅ |
| 4.1 | Auto-load `.env` via python-dotenv | ✅ |
| 5 | Multi-turn REPL | ✅ |
| 6.1 | Sliding-window compaction (user-turn aligned) | ✅ |
| 6.2 | LLM-summarization of dropped messages | ✅ |
| 7 | Session persistence to JSONL | ✅ |
| 7.1 | Metadata persistence (title, created_at, updated_at, cursor) | ✅ |

## Reference

- Upstream: `D:\CODE\nano\nananobot` (mirror of `github.com/anomalyco/opencode`)
- Focus subsystem: `nanobot/agent/` (loop.py, runner.py, memory.py)
- See upstream `AGENTS.md` for the project it descends from.