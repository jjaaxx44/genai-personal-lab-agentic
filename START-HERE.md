# Bundle note

This is scaffolding for a new repo, not part of the RAG repo it currently sits inside. Delete this file once you've done the moves below — everything else here is real repo content.

## Moving it out

```sh
mv ~/Desktop/genai-personal-lab/genai-personal-lab-agentic ~/Desktop/genai-personal-lab-agentic
cd ~/Desktop/genai-personal-lab-agentic
git init
cp .env.example .env          # then fill in at least one provider key
```

`taxonomy/` and `.claude/` are gitignored, exactly as in the RAG repo, so the plan, the checklist and the skills stay local.

## What's in it

| Path | What it is |
|---|---|
| `CLAUDE.md` | Working rules. Ten of them; rules 3–6 (budgets, trajectory, sandbox, tool errors) are the ones specific to agents |
| `DESIGN.md` | The design decisions: grounding, vocabulary, validated tokens, page patterns, house rules |
| `taxonomy/plans/AGENTIC_IMPLEMENTATION_PLAN.md` | The plan: stack, layout, contracts, pins, 16 steps with "done when" lists. Session one reads this |
| `taxonomy/genai-capability-taxonomy.md` | Your master checklist, copied across. Section 2 is this repo's scope |
| `README.md` | The repo's public README, written for someone arriving on GitHub |
| `pyproject.toml`, `Dockerfile`, `docker-compose.yml`, `.env.example` | Ready to build, except the one unpinned dependency Step 0 resolves |
| `.streamlit/config.toml` | The theme, with the tokens from `DESIGN.md` already in it |
| `.claude/skills/` | `app-ux-design` and `langfuse`, copied from the RAG repo |
| `app/static/fonts/` | Inter + JetBrains Mono, with the OFL notice. Instrument Serif is not here — this app has no serif |
| `samples/chinook.db` | For the SQL analyst demo |
| `samples/corpus/` | Empty except for a note. Put real documents here before Step 1 |
| `LICENSE` | MIT, with the RAG repo's AGPL/PyMuPDF notice removed — there's no PDF library in this one |

`app/core/` and `app/demos/agents/` are empty directories; Step 0 fills them.

## Decisions already made

- **16 demos.** All of taxonomy § 2 except the two already struck there, minus browser automation and coding agents, both deferred with reasons recorded in the plan.
- **Own identity.** Cool graphite surfaces, violet accent, mono headings, square corners — deliberately not the RAG lab's warm paper and pine green. The tokens were computed and checked: contrast ratios are in `DESIGN.md`, the five categorical colours stay separable under deuteranopia and protanopia (worst-case ΔE 25.8 light / 28.4 dark), and both sequential ramps step evenly in lightness.
- **Same infrastructure.** `app` + `mongodb` compose, same pins, same provider fallback chain. MongoDB now also holds LangGraph checkpoints and long-term memory.
- **No PDFs, no Neo4j, no PyMuPDF.** Input is a task; documents live in `samples/corpus/` as plain text.

## Open decisions, deliberately left for the step that needs them

- `deepagents` library vs hand-rolling the deep-agent ingredients (Step 12).
- `a2a-sdk` vs implementing the protocol directly (Step 11).
- Whether a run-id package earns its place at all (Step 0).

Each is written up in the plan under *Pinned versions* with the trade-off, so the decision gets made once, in the open.

## First session

Start it with something like:

> Read CLAUDE.md and taxonomy/plans/AGENTIC_IMPLEMENTATION_PLAN.md. Implement Step 0 only, then stop for review against its "done when" list.

Step 0 also has to resolve `langgraph-checkpoint-mongodb` and write the version back into both `pyproject.toml` and the plan — the pin is left blank on purpose rather than guessed.

One thing worth doing before Step 1: put a few real documents in `samples/corpus/`. Most of these demos are only interesting when the tools have something substantial to dig through, and the research and evaluation demos get much better if two documents in there disagree with each other.
