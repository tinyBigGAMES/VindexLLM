# VindexLLM — Anchored Development Process

## Purpose

This document defines the **3-level anchored workflow** used across all VindexLLM
development sessions. It prevents results drift — the slow corruption of working
code and decisions when an AI fills in gaps from inference rather than from what
was already decided.

Based on: Bill Burdick, "Your AI Keeps Forgetting What You Decided: Avoiding
Unanchored Results" (April 2026).

**Rule: Any feature or decision that only exists in the results (code) and not in
a spec or intent document is _unanchored_ and at risk of drift on the next edit.**

---

## The Three Levels

### Level 1 — Specs (Human-owned)

**Location:** `.claude\specs\`
**Owner:** You (the developer). Co-written with AI, but audited and owned by you.

A spec describes **what** you want and **why** you want it. The "why" is critical:
a capable AI that misunderstands your motivation will make internally consistent
_wrong_ choices across the whole project. Smarter AI needs more context about your
motivations, not less, because it infers more aggressively from whatever context
it has.

Each spec file covers one major feature area or phase. Contents:

- What you're building and who it's for
- What it should contain or accomplish
- Why — your motivations, goals, and values
- Design language, tone, constraints
- Decisions log: what you've already settled and why

**Existing master spec:** `.claude\tasks\TASK-DESIGN.md` — the project-wide spec
covering architecture, principles, build phases, and hardware targets.

**Per-phase specs** go in `.claude\specs\` as needed (e.g., `SPEC-ATTENTION.md`,
`SPEC-QUERY-ENGINE.md`).

### Level 2 — Intent (AI-written, human-reviewed)

**Location:** `.claude\intent\`
**Owner:** AI writes it, you review and approve before results change.

An intent document records how the AI interpreted the spec, what it plans to build,
and a manifest tracking completion status. This is the layer that prevents drift.

Each intent file corresponds to a spec (or a section of a spec). Contents:

- How the AI interpreted the spec
- Structure and layout of the results (the index into code)
- Bite-sized chunks the AI can navigate without reading everything
- Requirements manifest with completion status (checkboxes)
- Traceability back to specs (which spec section drives each item)

**The manifest** uses checkboxes:
- `[x]` — Done and verified
- `[ ]` — Not started or needs work
- `[~]` — Partially done / in progress
- `[!]` — Blocked or needs human decision

When a spec changes, the AI marks affected manifest items as `[ ]` (needs work)
so both you and the AI can see what needs updating.

**Why this matters:** Without intent docs, the AI re-reads results every time and
re-derives what it thinks the intent was. Each re-derivation introduces small
differences. Over multiple edits, those differences compound into drift. The intent
document is shorter and easier to review than the full results.

### Level 3 — Results (The actual code)

**Location:** `src\`, `shaders\`, `testbed\`
**Owner:** Generated from the intent document, not from scratch each time.

The results are the Delphi units, GLSL shaders, and test code. Each unit is
already a small, self-contained file — VindexLLM's per-unit architecture maps
naturally to the "small indexed chunks" principle.

**Traceability in results:** Use Delphi comments to link code back to intent:
```delphi
// INTENT: INTENT-ATTENTION.md #qk-norm — QK-norm per head after projection
```

This lets gap analysis detect drift between what was planned and what exists.

---

## Change Flow

The normal flow for making changes:

```
1. You update a spec          (.claude\specs\SPEC-XXX.md)
2. AI updates intent doc      (.claude\intent\INTENT-XXX.md)
   - Marks affected manifest items as [ ] (needs work)
   - Proposes what will change in results
3. You review the intent      (approve / correct / reject)
4. AI implements changes      (src\, shaders\, testbed\)
5. AI marks manifest items    [x] when done
6. SESSION.md updated         (current state for next session)
```

### Shortcuts (when you skip steps)

In practice, you'll sometimes make a quick change directly to results without
updating specs or intent first. That's fine — but unanchored changes accumulate.

When this happens, ask the AI to run **gap analysis**: it reads the results, compares
them against the intent docs, and anchors any untracked changes back into Levels 1
and 2. This catches drift before it compounds.

### When to skip the intent step

Small, surgical changes to a single routine that don't touch interfaces, don't add
dependencies, and don't change behavior visible to other units — these can go straight
to results. The AI should note the assumption: "Surgical change, no intent update
needed because [reason]."

---

## Gap Analysis

Gap analysis detects three kinds of problems:

1. **Drift** — Results have diverged from what the intent document describes.
2. **Unimplemented** — Intent manifest has items marked done but code doesn't match.
3. **Unanchored** — Code exists that isn't tracked in any intent document.

To run gap analysis, ask: "Run gap analysis on [unit/area]." The AI will:
1. Read the relevant intent document
2. Read the corresponding results (code)
3. Report any gaps, drifts, or unanchored changes
4. Propose updates to specs/intent to re-anchor everything

---

## Session Survival

This is the most urgent practical benefit. When a session ends (context limit,
crash, new day), the next session picks up from:

1. `SESSION.md` — current state, what's done, what's next
2. `CLAUDE.md` — this file, the process itself
3. `.claude\intent\` — the manifest shows what's complete and what needs work
4. `.claude\specs\` + `TASK-DESIGN.md` — the authoritative specs

The AI reads these small docs instead of the entire codebase. The intent docs
index into the code so it only reads the units it needs to change.

**Session startup sequence:**
1. Read `SESSION.md` (current state)
2. Read `CLAUDE.md` (this process)
3. Read relevant intent docs (manifest = what's active)
4. Read relevant specs only if working on a new area
5. Read code only for the specific units being changed

---

## Git Integration

All three levels are version-controlled. Git is the safety net.

- **Git binary:** `C:\Users\tinybiggames\AppData\Local\GitHubDesktop\app-3.5.7\resources\app\git\cmd\git.exe`
- **Before risky changes:** Commit current state so you can recover
- **Commit style:** Short single-line messages
- **What to commit:** Specs, intent docs, results, and session state together when they change as a unit

---

## File Map

```
VindexLLM\repo\
├── CLAUDE.md                        ← This file (process definition)
├── .claude\
│   ├── soul\
│   │   └── SESSION.md               ← Session state (read first every session)
│   ├── tasks\
│   │   └── TASK-DESIGN.md           ← Master project spec (Level 1)
│   ├── specs\
│   │   └── SPEC-*.md                ← Per-feature specs (Level 1)
│   ├── intent\
│   │   └── INTENT-*.md              ← AI intent docs + manifests (Level 2)
│   ├── reference\                   ← Reference material (Vulkan headers, etc.)
│   ├── scripts\                     ← Build/utility scripts
│   └── tools\                       ← Standalone tools (glslangValidator, etc.)
├── src\                             ← Delphi source (Level 3 results)
├── shaders\                         ← GLSL compute shaders (Level 3 results)
└── testbed\                         ← Test harness (Level 3 results)
```

---

## Naming Conventions

- **Specs:** `SPEC-<AREA>.md` (e.g., `SPEC-ATTENTION.md`, `SPEC-QUERY-ENGINE.md`)
- **Intent:** `INTENT-<AREA>.md` (e.g., `INTENT-ATTENTION.md`)
- **Spec and intent names match** for traceability

---

## Bootstrapping Existing Work

VindexLLM is mid-project (Phases A-C complete, attention debugging in progress).
The existing `TASK-DESIGN.md` serves as the master spec. To anchor existing work:

1. **Don't start over.** The existing code is the ground truth.
2. **Extract intent retroactively.** When starting work on an area, create an
   intent doc from the existing code and SESSION.md state. Review it.
3. **Add traceability incrementally.** Add `// INTENT:` comments to code as
   it gets touched, not all at once.
4. **Create specs on demand.** Write a per-feature spec when a feature area
   needs deeper "what and why" than TASK-DESIGN.md provides.

The process doesn't require rewriting anything. It layers on top of what exists.

---

## Quick Reference for AI

**Before changing results (code):**
1. Is there an intent doc for this area? If not, create one first.
2. Update the intent doc to reflect the planned change.
3. Present the intent update for review.
4. Only after approval, implement the change.
5. Mark manifest items as done.

**When a spec changes:**
1. Read the changed spec.
2. Update the intent doc — mark affected manifest items as `[ ]`.
3. Present the updated manifest for review.
4. Implement approved changes.

**When asked to run gap analysis:**
1. Read intent doc + corresponding code.
2. Report: drifts, unimplemented items, unanchored code.
3. Propose re-anchoring into specs/intent.
