# GenAI Agentic Lab

A Streamlit app where each page demonstrates one agentic or multi-agent technique, 16 in total. Demo/POC quality — the point is to show the technique and the reasoning, not to ship production software.

Sibling repo: `genai-personal-lab-rag`, same method, same house rules, different subject. Nothing is imported across the two; patterns are copied deliberately, not shared.

**Read `taxonomy/plans/AGENTIC_IMPLEMENTATION_PLAN.md` before starting a step.** It holds the stack, exact versions, repo layout, the shared `run()` / `resume()` contract, the step order and each step's "done when" list. Don't restate it here; if this file and the plan disagree on technical detail, the plan wins.

Capability checklist: `taxonomy/genai-capability-taxonomy.md` § 2 — tick a demo off when it lands.

> `taxonomy/` is gitignored on purpose: the plan and the checklist are working notes and stay in the local copy. If you cloned this repo, those two paths won't exist for you, and nothing in `app/` depends on them.

UI work: `DESIGN.md` holds this app's design tokens and page patterns. Method comes from the `app-ux-design` skill; Streamlit mechanics from the `developing-with-streamlit` skill that ships inside the Streamlit package.

## Rules

1. **Standalone demos.** Each demo is `app/demos/agents/<name>/` (`agent.py`, `page.py`, `README.md`), imports from `app/core/` only, never from another demo, and touches only its own MongoDB collections. Duplication between demos beats shared abstractions, and demos are never refactored together "for consistency".
2. **Thin core.** Something moves into `core/` when a *second* demo needs it, not in anticipation. The toolbox (`core/tools.py`) is the one deliberate exception: every demo needs the same tools, and one demo's subject *is* tool design — so the tools are shared and the way each demo wires them up is not.
3. **Every run is bounded.** No agent loop ships without a step cap, a token budget and a wall-clock deadline, all read from config and all visible in the UI. An agent that hits a cap stops and says which cap it hit. This is a hard rule, not a nice-to-have: an unbounded loop on a free tier is how you lose an afternoon and a quota.
4. **The trajectory is the product.** Every demo records each step as thought / action / observation / decision, with its tool, arguments, tokens and latency, and renders it. An agent that produces a good answer with an unreadable trajectory has failed the demo.
5. **Sandboxed tools only.** Tools read from the bundled corpus, the bundled SQLite (read-only, SELECT only), an allow-listed web search, and a virtual file system under `data/`. No shell, no arbitrary HTTP, no writes outside `data/`, no credentials in tool arguments. A demo that needs a new tool adds it to `core/tools.py` behind the same guards.
6. **No automated tests.** Verification is manual: run the app, walk the step's "done when" list.
7. **Fixed stack.** Never add, remove or bump a dependency, service or local model, or change a pin, while implementing a step — raise it as a decision first. Everything has to keep running on 4 CPUs / 8 GB, CPU only, no GPU.
8. **Import discipline.** Never import `langchain_community` or `langchain_classic` — both are archived/legacy and are present only as transitive dependencies, so the import will work and nothing will warn you. The ReAct demo has no `langchain` imports at all; its loop is hand-written so the mechanics stay visible.
9. **Teaching value first.** Each page shows the graph, the trajectory, the budget spent, and every decision the agent made with the reason it gave. Code that hides the technique is a defect.
   Every README follows one fixed structure, in this order: `What it is` (the technique in general — how it works and why, never this repo's implementation), `Control flow` and `State and memory` (one mermaid diagram each; `st.markdown` renders mermaid fences natively, and so does GitHub), `Strengths`, `Limitations` (inherent trade-offs *and* the ways it fails, merged — not a separate failure-modes section), `Where to use it`, and a closing `In this demo` holding every implementation specific: libraries, collection names, env vars, caps, caveats.
10. **Soft failure.** Rate limits, a paused database, a missing optional key and a tool that throws all give a clear UI message — never a stack trace, never a crash on startup. A tool error is fed back to the agent as an observation, because recovering from one is part of what these demos teach.

## How to work

One step from the plan per session, in order. Implement it, then stop so a human can check it against that step's "done when" list. If a step turns out to be wrong or impossible, say so and propose the change instead of working around it — and once agreed, update `AGENTIC_IMPLEMENTATION_PLAN.md` in the same session, since every later step reads from it.
