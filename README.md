# GenAI Agentic Lab

Sixteen agentic and multi-agent techniques, one Streamlit app, one page each. Every page gives an agent a real task, real tools and a real budget, then shows its working: the control flow it took, every step it decided on, every tool call and what came back, and what the run cost in steps, tokens and seconds.

It is a teaching repo. Each demo is written to make its technique legible, not to be reused — so the code is deliberately repetitive, there is no shared abstraction layer, and there are no tests. The point is that you can read one folder top to bottom and understand what the technique actually does.

Everything runs on a laptop: 4 CPUs, 8 GB of RAM, no GPU. Embeddings are a local model; only the LLM calls and the web search leave the machine.

Sibling repo: [genai-personal-lab-rag](https://github.com/jjaaxx44/genai-personal-lab-rag) — sixteen RAG techniques, same method, same house rules.

## The demos

The home page is organised around symptoms rather than names — you pick the way your agent is failing, and it points you at the techniques that address it. The full catalogue:

### The single-agent loop
The basic loop, the interface it acts through, and the two ways of making it deliberate.

| # | Technique | What it does |
|---|---|---|
| 01 | ReAct | Think, act, look at what came back, think again — written out by hand as a prompt and a for-loop. |
| 02 | Tool design | The same task against a badly described toolset and a well described one, side by side. |
| 03 | Plan-and-Execute | A planner writes the whole plan first; an executor works through it and a replanner revises it. |
| 04 | Reflexion | The work is graded against the task, and a written critique is fed into the next attempt. |

### State, control and the human
What the agent remembers, when it must stop and ask, and how it hands work back.

| # | Technique | What it does |
|---|---|---|
| 05 | Agent memory | Three tiers at once: the trimmed message window, the resumable thread, and facts recalled from past runs. |
| 06 | Human-in-the-loop gates | The run pauses before a consequential action and resumes from its checkpoint once you approve, edit or reject it. |
| 07 | Escalation and handoff | The agent decides it cannot finish, and writes a packet saying what it tried and what it needs from you. |

### More than one agent
Four ways of splitting work between agents, and what each costs.

| # | Technique | What it does |
|---|---|---|
| 08 | Sub-agent delegation | Work goes to a sub-agent with its own context and budget, and only a summary comes back. |
| 09 | Supervisor–worker | One agent routes each turn to a specialist and decides when the task is done. |
| 10 | Swarm | No supervisor: peers hand the task to each other, and whoever holds it owns it. |
| 11 | Agent-to-agent (A2A) | Two agents that share nothing talk over a protocol: discovery, a task, status updates, an artefact. |

### Long horizon
Agents that run for a while, and the ways that goes wrong.

| # | Technique | What it does |
|---|---|---|
| 12 | Deep agents | An explicit to-do list, a virtual file system, sub-agents and a prompt that ties the three together. |
| 13 | Autonomous goal loop | The agent sets its own next objective until it argues itself to a stop — or the budget stops it. |
| 14 | Deep research | A question is decomposed, searched, read and synthesised into a brief that cites its sources. |

### Applied, then measured
| # | Technique | What it does |
|---|---|---|
| 15 | Text-to-SQL analyst | The agent reads a schema, writes SQL, runs it read-only, and queries again when the rows don't answer the question. |
| 16 | Agent evaluation | Outcome scored separately from trajectory, so a right answer reached by a wasteful route is visibly worse. |

Each demo folder has a `README.md` covering the technique in general — how it works, what it is good at, how it fails, where to use it — followed by an `In this demo` section with every implementation detail. The diagrams are mermaid, so they render on GitHub and inside the app.

## What every demo has

- **A budget.** Steps, tokens and wall clock, all capped, all displayed while the run happens. A run that hits a cap stops and names it. Unbounded agent loops are the single most expensive mistake in this space, so no demo here can make one.
- **A trajectory.** Every thought, action, observation, decision, gate and handoff is recorded as a step and rendered, including the ones that failed.
- **A sandbox.** Tools read a bundled text corpus, a bundled SQLite database (SELECT-only, row-capped), an allow-listed web search, and a per-run directory on disk. There is no shell tool.
- **Soft failure.** A tool that errors hands the error back to the agent as an observation, because recovering from one is part of what these demos teach.

## Stack

| Layer | Choice |
|---|---|
| UI | Streamlit multipage app, Python 3.12 |
| LLM | LangChain chat models chained with `.with_fallbacks(...)`: Gemini → Groq → OpenAI → local Ollama. A provider joins the chain only when its key *and* its model are set. |
| Orchestration | LangChain v1 and LangGraph, except ReAct, which is plain Python so the mechanics stay visible |
| Durable state | `langgraph-checkpoint-mongodb` — gates and long runs resume after a rerun or a restart |
| Memory + corpus index | MongoDB Atlas Local, in Docker — no cloud database account needed |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers — local, CPU |
| SQL | SQLite (the bundled Chinook database) via SQLAlchemy, read-only |
| Web search | `ddgs` (no API key) |
| Evaluation | LLM-as-judge and trajectory metrics, with Ragas where a reference answer exists |
| Tracing | Langfuse — optional, and silently disabled when keys are missing |

Every direct dependency is pinned in [pyproject.toml](pyproject.toml), with a committed `uv.lock` for the transitive ones. Torch comes from the CPU-only index; the PyPI Linux wheels would pull in several GB of CUDA packages that this app has no use for.

## Running it

**Prerequisites:** Docker (with Compose). Optionally [Ollama](https://ollama.com) on the host if you want a local LLM as the last fallback.

1. Copy the environment template:

   ```sh
   cp .env.example .env
   ```

2. Fill in at least one LLM provider key. Gemini and Groq both have usable free tiers, and the defaults in the template point at their cheapest models. Change `MONGODB_PASSWORD` from `change-me`, and make `MONGODB_URI` match it.

   Agent loops make many calls per run, so free-tier rate limits bite harder here than in a RAG app — two configured providers is worth it. Langfuse is optional.

3. Start it:

   ```sh
   docker compose up --build
   ```

   Then open <http://localhost:8503>. (The app serves on 8501 inside the container; 8503 is published on the host so this lab can run alongside the RAG lab, which uses 8501.) First boot downloads the embedding model into a cached volume, so it is slower than later ones.

Put any `.md` or `.txt` files you want the agents to be able to search into [samples/corpus/](samples/corpus/); they are indexed on first use.

Note that application code is **not** bind-mounted into the container — only `./data` and the model cache are. A code change needs `docker compose up --build`, not a restart. Streamlit's file watcher is switched off in [.streamlit/config.toml](.streamlit/config.toml) for the same reason.

## Layout

```
├── app/
│   ├── Home.py                  # symptom grid + full catalogue
│   ├── core/                    # config, LLM, embeddings, tools, corpus, Mongo, checkpoints, budget, tracing, shared UI
│   ├── static/fonts/            # the two typefaces the theme loads
│   └── demos/agents/<name>/     # agent.py, page.py, README.md — one folder per technique
├── samples/
│   ├── corpus/                  # what the search tool reads
│   └── chinook.db               # the SQL analyst's database
├── .streamlit/config.toml       # theme tokens and server settings
├── docker-compose.yml           # app + mongodb
├── Dockerfile
├── DESIGN.md                    # design tokens and page patterns
└── CLAUDE.md                    # working rules for this repo
```

A demo imports from `app/core/` and never from another demo, and touches only its own MongoDB collections. Something moves into `core/` when a second demo needs it, not in anticipation — duplication between demos is the intended trade-off, because it keeps each one readable on its own. The one deliberate exception is the toolbox: every demo needs the same tools, so they are shared, and what differs is how each demo describes and wires them up.

## Licence

MIT — see [LICENSE](LICENSE). Unlike the RAG lab, this repo has no AGPL-licensed dependency, because there is no PDF library in it. Bundled assets keep their own licences: the typefaces in [app/static/fonts/](app/static/fonts/) are SIL Open Font License 1.1, and `samples/chinook.db` is the [Chinook sample database](https://github.com/lerocha/chinook-database), MIT.
