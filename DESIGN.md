# Design decisions — GenAI Agentic Lab

The values and patterns this app is built from. Method lives in the `app-ux-design` skill; Streamlit mechanics live in the `developing-with-streamlit` skill that ships inside the Streamlit package. This file holds what is specific to this repo — and the reason for each choice, so later screens can be checked against intent rather than taste.

This is a sibling of the RAG lab, not a reskin of it. The method is the same and the house rules are the same; the grounding, the palette, the type pair and the signature element are all different, because the subject is different. If a screen here could be dropped into the RAG app without anyone noticing, something has gone wrong.

## Grounding

The subject is **control flow**: a model decides what to do next, does it, looks at what came back, and decides again — branching, retrying, handing off, sometimes stopping to ask a human. Unlike retrieval, there is no fixed chain of stages; the shape of a run is only known once it has run. The references the design borrows from:

1. **Mission-control and flight-deck panels** — a bounded budget on permanent display, discrete statuses, a gate that halts the sequence until a human signs it off.
2. **State machines and transit diagrams** — nodes and edges drawn in full, with the route actually taken picked out against the routes that weren't.
3. **Dispatch logs** — sequential, timestamped entries: who acted, what they observed, what they decided, what it cost.

So: the graph and the trajectory are the hero, the budget is always visible, and the chrome behaves like a panel — quiet, square, uniform.

**The first screen shows failure symptoms**, not a welcome banner and not sixteen technique names.

## Vocabulary

One word per concept, everywhere: UI labels, READMEs, log lines, variable names.

| Use | Not |
|---|---|
| task | query, goal, prompt, request |
| run | session, execution, invocation |
| step | iteration, turn, cycle |
| trajectory | history, log, transcript, scratchpad |
| tool | function, action, skill, capability |
| observation | tool output, result, response |
| decision | reasoning, thought process, rationale |
| gate | approval, checkpoint, HITL, interrupt |
| handoff | delegation, transfer, routing |
| budget | limit, quota, guardrail |
| agent | bot, assistant, model, worker (except in supervisor–worker, where `worker` is the subject) |
| demo | example, module, app |

`thought` survives in one place only: the ReAct demo, where the thought/action/observation triple is the subject.

## Identity, with reasons

- **Concept:** an instrument panel for a process that runs on its own — the app exists to show *what the agent decided and what it spent*, so the graph, the trajectory and the budget get the visual weight.
- **Surface temperature:** cool graphite (dark) / cool paper-white (light). Reason: the panel reference, and deliberate distance from the RAG lab's warm paper — the two apps must not read as one.
- **Accent with one job:** violet marks *the path actually taken* — the visited node, the current step, the primary button. Nowhere else, so its presence always means "this happened".
- **Type pair:** monospace headings (panel labelling), sans body (legibility), monospace for every number and identifier. The RAG lab pairs a serif with its notebook grounding; a panel is labelled in the same face it reports in, so headings and readings share JetBrains Mono and the serif is gone.
- **Edges:** 1px borders, **no radius at all**, no shadows, no gradients. Reason: housing, not cards. The RAG lab uses a small radius; this one is square, which is the fastest visible tell between the two apps.
- **Signature element:** the **trajectory track** — a numbered rail down the page, one row per step, retries stacked under the step they replace and handoffs indented under the agent that made them.

**Considered and rejected:** amber-on-graphite (the obvious mission-control move, but amber is a warning colour and it would fight every degraded-state message the app has to show); reusing the RAG lab's pine green (two labs would become indistinguishable, and green reads "all clear" on a panel where the point is that things branch and fail).

**Known adjacency:** violet-on-graphite sits near the "AI startup dark mode" cliché. What keeps it clear: no gradients, no glow, no rounded cards, a mono heading face rather than a geometric sans, and an identity that rests on the trajectory track rather than on the background. If a screen starts to feel like a landing page, that's the drift to correct.

## What structure encodes

Structural devices carry meaning here; none are decorative.

| Device | Means |
|---|---|
| Row order down the track | execution sequence |
| Row indent | a sub-agent's step, under the agent that delegated it |
| Rows stacked with a shared index (`04`, `04′`) | a retry replacing an attempt |
| Accent outline on a graph node | visited on this run |
| Muted graph node | reachable, not taken |
| Rule across the track | a gate: everything above it is committed, nothing below has run |
| Mono type | a value you can compare with another value |
| Bar length in the budget strip | fraction of a cap consumed |

**Motion:** exactly one moment — rows appearing on the track as steps complete, and the budget bars advancing with them. Everything else is still.

## Tokens

All values below were computed and checked: contrast against the surface they sit on, monotonic lightness through the sequential ramps, and separation between categorical series under normal vision, deuteranopia and protanopia. The numbers quoted are the measured ones. Don't substitute by eye; re-validate if you change one.

### Light

| Token | Value | Contrast vs `bg` |
|---|---|---|
| `bg` | `#F5F6F8` | — |
| `surface` (sidebar, code, raised) | `#E9EBF0` | 1.10 |
| `border` | `#D2D6DF` | 1.35 |
| `text` | `#14161B` | 16.7 |
| `text-muted` | `#5E6570` | 5.4 |
| `accent` (fills, visited, active) | `#4B3FC7` | 6.9 (white on it: 7.4) |
| `accent-text` (links, accent text) | `#443AB8` | 7.6 |
| `warning` | `#8A5A00` | 5.5 |
| `error` | `#B22F2A` | 5.8 |

### Dark

| Token | Value | Contrast vs `bg` |
|---|---|---|
| `bg` | `#141519` | — |
| `surface` | `#1D1F26` | 1.11 |
| `sidebar-bg` | `#0E0F12` | 1.05 |
| `border` | `#2E323B` | 1.42 |
| `text` | `#E7E9EE` | 15.0 |
| `text-muted` | `#949AA6` | 6.5 |
| `accent` | `#8E82F5` | 5.8 |
| `warning` | `#E0A94A` | 8.6 |
| `error` | `#E4736B` | 6.1 |

### Data colours

Categorical, in fixed order (agent or run comparisons on the evaluation page). Series index means the same hue in both modes: violet, amber, teal, sky, brown.

- light: `#4B3FC7`, `#9A6400`, `#1E8A78`, `#1E7FB8`, `#6F4A2E` — worst-case separation between any two, across normal, deutan and protan vision: ΔE 25.8
- dark: `#8E82F5`, `#C9992F`, `#6FD0BC`, `#57B6DE`, `#B58763` — worst case ΔE 28.4

Sequential (step cost, confidence, budget consumption), low → high:

- light: `#C2B4E1`, `#AF9CDC`, `#9D84D8`, `#8B6AD7`, `#7850D6`, `#622FD7`, `#4C1DBA`, `#391292`, `#270A6A`, `#170442`
- dark: `#291657`, `#3E237E`, `#5431A4`, `#6A44C3`, `#8061C8`, `#957DCE`, `#A996D5`, `#BFB1DE`, `#D4CCE7`, `#EAE6F3`

Both ramps step evenly in lightness (~7.8 L* per step in light, ~8.7 in dark) and are monotonic end to end, so a value's position in the ramp is readable without the legend.

### Type and space

- Headings: JetBrains Mono (500); body: Inter at 15px; numbers, IDs, code: JetBrains Mono (400).
- Self-hosted from `app/static/fonts/` with `[[theme.fontFaces]]` and `server.enableStaticServing = true`, so there's no font-CDN dependency at runtime.
- Spacing scale: 4 / 8 / 16 / 24 / 48. Nothing in between.
- Reading measure 60–80 characters; wide screens get margins, not longer lines.
- Numbers: fixed precision with units — `12.4 s`, `7/12 steps`, `3,480 tokens`, `run 9f21c4…`.

Everything above is set in `.streamlit/config.toml` as theme tokens — never CSS injection, never colour literals in page code.

## Page patterns

### Home — symptom first, catalogue behind one click

The entry surface is the **symptom grid**: bordered tiles, three to a row, each stating a way an agent breaks, in the reader's own words — never a technique name.

```
It keeps calling tools and never stops.     It picks the wrong tool, or calls
                                            the right one with junk arguments.

`13` Autonomous loop  `04` Reflexion        `02` Tool design  `01` ReAct
```

The symptom is body text; the demos it routes to are tertiary buttons carrying a mono index. The grid's last tile is the way in for a reader with no symptom yet — `Nothing yet — show me the basic loop` — and holds the page's **one** accent button, `Open ReAct`.

The reason this beats an index: every technique here exists because a plain ReAct loop broke in a particular way, so the break is the thing a reader recognises.

Below the grid, one full-width bordered button, `Show all sixteen techniques`, toggles the **full catalogue** inline: the same card as the symptom tile, three to a row, grouped under family headings with a thin rule. Each card carries a tertiary button holding mono index and name, one line on what the technique does, and the **shape of its control flow** (`plan → act → observe ↻`, `supervisor → worker ×3 → merge`, `act → gate ⏸ → act`), taken from that demo's own `SHAPE`.

The shapes are the reason the catalogue is worth showing, and why the cards share a width: the shapes line up, so the control flows visibly diverge card to card — Plan-and-Execute has no loop back to the planner, Reflexion loops through a critic, HITL stops dead in the middle. A card grid is the default that makes apps interchangeable, so it is used here on purpose and only here: the grid is a **comparison**, not a menu of tiles.

Both surfaces read from one `DEMOS` dict keyed by index, so a name or shape is written once.

**No card or tile ever scrolls its own text**, so neither carries a `height` at all — `st.container(border=True)` and nothing more. Each one is exactly as tall as its own content. (The RAG lab learned this the hard way: any `height`, including `"stretch"`, turns a container into a scroll surface and hides the very thing the card exists to show.)

Home is width-constrained and centred. The app is `layout="wide"` for the demo pages, where the trajectory and the trace genuinely use the width; a page of sentences gets margins instead of 120-character lines.

### Sidebar navigation

Built by hand in `sidebar_nav()`: a `Home` page link, then a **collapsed expander** labelled `Demos` holding all sixteen `st.page_link`s. Streamlit's own navigation is turned off with `st.navigation(..., position="hidden")`.

The reason is the demo page's sidebar, not the navigation: the task presets, the demo's own settings, the budget caps and `Clear my data` live there, and an open list of sixteen pushes all of them below the fold.

**`expanded=` on `st.navigation` cannot do this.** The first time anyone clicks "View N more", Streamlit's frontend writes `sidebarNavState=expanded` into `localStorage` and force-expands the list on every page from then on, ignoring whatever `expanded=` says — and the flag is per origin *including port*, so the same build looks correct on one port and wrong on another. The expander is keyed on the current page (`key=f"nav_demos_{current.title}"`), so every navigation gives it a fresh identity and arriving at a demo always finds the list closed.

`sidebar_nav()` is called between `st.navigation()` and `nav.run()`, so it sits above whatever sidebar content the page itself adds.

### Demo page

1. **Header** — demo name (mono) + one plain sentence on what the technique does.
2. **Graph map** — the demo's control flow drawn in full, muted, before anything runs. It is the empty state and the explanation at once; during a run, visited nodes take the accent outline and the current one is filled. This is the agentic counterpart of the RAG lab's pipeline ribbon: a ribbon assumes a fixed chain, and there isn't one here.
3. **Task row** — input + `Run`, wrapped in a form so Enter submits it. A `st.pills` row of two or three preset tasks sits above it, because a reader shouldn't have to invent a task that exercises the technique.
4. **Budget strip** — steps used / cap, tokens / cap, elapsed / deadline, tool calls. Always visible, mono, with a bar from the sequential ramp. It sits *above* the trajectory, not in a metrics row at the bottom, because with agents the cost is the thing that surprises people.
5. **Trajectory track** — one bordered row per step as it happens: mono index, kind chip (`think` / `act` / `observe` / `decide` / `gate` / `handoff`), the tool and its arguments, the observation, and the step's own tokens and latency. Arguments and long observations collapse into an expander; nothing is truncated without a way to open it. Retries stack under the step they replace with a primed index (`04`, `04′`). Sub-agent steps are indented under the handoff that spawned them.
6. **Gate panel** — only in demos that pause. A rule across the track, then the proposed action rendered in full, then `Approve` / `Edit` / `Reject` with a reason box. The run resumes from the checkpoint; it does not restart.
7. **Final output** — the answer or artefact, with the steps it rests on cited by index.
8. **Detail tabs** — `How it works` (renders the demo's README.md) first, then `Trace` (raw messages, tool JSON, checkpoint ids). The pair is stateful: `st.tabs([...], key=f"{DEMO}_tabs", on_change="rerun")`, so the selected tab lives in session state and the page can move it.

   The rule is **explain first, then get out of the way**. On arrival nothing has run, so the reader gets the explanation. The moment a run finishes, the page sets `st.session_state[f"{DEMO}_tabs"] = "Trace"` — set immediately after the result is stored, which is always above the `st.tabs` call, so the widget picks it up the same run.

   READMEs carry two mermaid diagrams (control flow, then state and memory). `st.markdown` renders ` ```mermaid ` fences natively, so the README stays one plain file that renders in the app and on GitHub. Diagrams are unstyled — no `classDef`, no colour literals — so they inherit the app theme in both modes. The one exception is the shared graph renderer, which has to override LangGraph's own hardcoded classDefs.

Sidebar: task presets, demo settings, budget caps, `Clear my data`.

### Shared behaviour

- Session-state keys are namespaced per demo (`f"{DEMO}_task"`), since state is shared across pages.
- `@st.cache_resource` for models and clients; `@st.cache_data` for derived data keyed by run.
- A run streams: steps appear on the track as they complete, inside a `st.status` with named sub-steps — never an anonymous spinner, and never a frozen page for thirty seconds.
- Errors follow the three-kind split: expected/actionable → inline warning, app keeps working; input rejected → name the limit and the actual value; unexpected → short message, traceback to the log. A tool error is *not* an app error: it goes onto the track as an observation and the agent gets to react to it.
- A run that hits a cap ends with a plain sentence naming the cap and its value, not an exception.
- Charts: evaluation page only, themed colours in fixed order, one measure per chart.
- Accessibility floor: validated contrast, real labels on every control, focus visible, meaning never carried by colour alone (kind chips and node states carry text too).

## House rules

No emoji (Material Symbols if an icon is genuinely needed). Sentence case. Buttons are verbs naming their object. Active voice, plain language, the vocabulary above. No custom CSS, no third-party components needing a Node build, no browser storage.
