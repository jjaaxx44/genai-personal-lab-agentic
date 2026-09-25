"""The entry surface, and the app's navigation.

Symptom first, catalogue behind one click (DESIGN.md). Every technique here
exists because a plain ReAct loop broke in a particular way, so the break is what
a reader recognises — not the name of the fix.

`DEMOS` is the single source for a demo's name, its one-line hook and the shape of
its control flow; the symptom tiles and the catalogue both read from it. It is
duplicated here rather than imported from the demo folders because a demo page is
a script: importing one would run it.
"""

import streamlit as st

st.set_page_config(page_title="GenAI Agentic Lab", layout="wide")

# index -> (name, what the technique does, the shape of its control flow).
DEMOS: dict[str, tuple[str, str, str]] = {
    "01": (
        "ReAct",
        "The loop itself, written by hand: think, act on a tool, read what came back, decide again.",
        "think → act → observe ↻",
    ),
    "02": (
        "Tool design",
        "One task against two toolsets, so a name, a schema and an error message show up as behaviour.",
        "act → observe ↻ ×2 toolsets",
    ),
    "03": (
        "Plan-and-Execute",
        "The plan is written first, executed step by step, and revised when a step invalidates it.",
        "plan → execute ×n → replan ↻",
    ),
    "04": (
        "Reflexion",
        "An attempt is scored against the task, critiqued in writing, and retried with the critique in hand.",
        "act → evaluate → reflect ↻",
    ),
    "05": (
        "Agent memory",
        "Three tiers at once: the message window, the checkpointed thread, and facts recalled from past runs.",
        "recall → act ↻ → write memory",
    ),
    "06": (
        "Human-in-the-loop gates",
        "The run stops before a write and waits for a person to approve, edit or reject it.",
        "act → gate ⏸ → act",
    ),
    "07": (
        "Escalation and handoff",
        "Below its own confidence threshold the agent stops and writes a handoff packet instead of guessing.",
        "act ↻ → judge → escalate ⏸",
    ),
    "08": (
        "Sub-agent delegation",
        "Bounded work goes to a sub-agent with its own context and budget; only a summary comes back.",
        "parent → sub-agent ⤵ → summary",
    ),
    "09": (
        "Supervisor–worker",
        "A supervisor routes each turn to one specialist and decides when the task is done.",
        "supervisor → worker ×3 ↻",
    ),
    "10": (
        "Swarm",
        "No router: peers hand the task to each other, and whoever holds it owns it.",
        "peer ⇄ peer ⇄ peer",
    ),
    "11": (
        "Agent-to-agent (A2A)",
        "Agents that share no memory cooperate over the protocol: cards, tasks, status, artefacts.",
        "discover → task → status ↻ → artefact",
    ),
    "12": (
        "Deep agents",
        "A to-do list held in state, a file system for notes and drafts, and sub-agents for the pieces.",
        "plan → write files ↻ → assemble",
    ),
    "13": (
        "Autonomous goal loop",
        "Goal in, objectives it sets itself, critique it writes itself — and a tally of how it starts to spin.",
        "objective → act → critique ↻",
    ),
    "14": (
        "Deep research",
        "Break the question up, search per part, take notes, then synthesise a brief with citations.",
        "decompose → search ×n → synthesise",
    ),
    "15": (
        "Text-to-SQL analyst",
        "Inspect the schema, write SQL, run it read-only, read the rows, then answer or query again.",
        "schema → SQL → rows → answer ↻",
    ),
    "16": (
        "Agent evaluation",
        "One task set through several demos, scored twice: on the answer, and on the route it took.",
        "run ×n → score → compare",
    ),
}

# The entry surface. Each tile is a way an agent breaks, in the reader's words --
# never a technique name. The last tile is the way in for someone with no symptom yet.
SYMPTOMS: list[tuple[str, list[str]]] = [
    ("It keeps calling tools and never stops.", ["13", "04"]),
    ("It picks the wrong tool, or calls the right one with junk arguments.", ["02", "01"]),
    ("It charges ahead on a long task with no plan, and drifts off what I asked.", ["03", "12"]),
    ("It forgets what I told it the moment the page reloads.", ["05", "06"]),
    ("It was about to write something, and nobody was asked first.", ["06", "07"]),
    (
        "One agent is doing five jobs badly, and its context fills up with work that isn't its own.",
        ["08", "09"],
    ),
    ("The work needs two specialists, and neither of them owns the task.", ["10", "11"]),
    (
        "The answer has to be computed from a database, or gathered from several sources with citations.",
        ["15", "14"],
    ),
    ("It reached the right answer by a route I cannot defend.", ["16", "07"]),
]

BASELINE_PROMPT = "Nothing yet — show me the basic loop everything else is a repair of."

# The families are the plan's phases: each one is a class of problem, in build order.
FAMILIES: dict[str, tuple[str, list[str]]] = {
    "The single-agent loop": (
        "One agent, one context, one budget — and the four things that make or break it.",
        ["01", "02", "03", "04"],
    ),
    "State, control and the human": (
        "What the agent remembers, where it has to stop, and when it should hand the task back.",
        ["05", "06", "07"],
    ),
    "More than one agent": (
        "Split the work between agents, and choose who decides what happens next.",
        ["08", "09", "10", "11"],
    ),
    "Long horizon": (
        "Tasks too long for one context window, and the failure modes that come with them.",
        ["12", "13", "14"],
    ),
    "Applied, then measured": (
        "One application end to end, and the measurement that says whether any of it worked.",
        ["15", "16"],
    ),
}

# Filled in one entry at a time, as each demo lands. A tile for a demo that has no
# page yet says so quietly rather than 404ing.
BUILT_PAGES: dict[str, st.Page] = {
    "01": st.Page(
        "demos/agents/react/page.py",
        title="01 ReAct",
        url_path="react",
        icon=":material/sync:",
    ),
    "02": st.Page(
        "demos/agents/tool_design/page.py",
        title="02 Tool design",
        url_path="tool_design",
        icon=":material/construction:",
    ),
    "03": st.Page(
        "demos/agents/plan_execute/page.py",
        title="03 Plan-and-Execute",
        url_path="plan_execute",
        icon=":material/checklist:",
    ),
    "04": st.Page(
        "demos/agents/reflexion/page.py",
        title="04 Reflexion",
        url_path="reflexion",
        icon=":material/replay:",
    ),
    "05": st.Page(
        "demos/agents/memory/page.py",
        title="05 Agent memory",
        url_path="memory",
        icon=":material/history:",
    ),
    "06": st.Page(
        "demos/agents/hitl/page.py",
        title="06 Human-in-the-loop gates",
        url_path="hitl",
        icon=":material/pan_tool:",
    ),
    "07": st.Page(
        "demos/agents/escalation/page.py",
        title="07 Escalation and handoff",
        url_path="escalation",
        icon=":material/support_agent:",
    ),
    "08": st.Page(
        "demos/agents/subagents/page.py",
        title="08 Sub-agent delegation",
        url_path="subagents",
        icon=":material/call_split:",
    ),
    "09": st.Page(
        "demos/agents/supervisor/page.py",
        title="09 Supervisor-worker",
        url_path="supervisor",
        icon=":material/account_tree:",
    ),
    "10": st.Page(
        "demos/agents/swarm/page.py",
        title="10 Swarm",
        url_path="swarm",
        icon=":material/hub:",
    ),
    "11": st.Page(
        "demos/agents/a2a/page.py",
        title="11 Agent-to-agent (A2A)",
        url_path="a2a",
        icon=":material/swap_horiz:",
    ),
    "12": st.Page(
        "demos/agents/deep_agent/page.py",
        title="12 Deep agents",
        url_path="deep_agent",
        icon=":material/folder_open:",
    ),
    "13": st.Page(
        "demos/agents/autonomous/page.py",
        title="13 Autonomous goal loop",
        url_path="autonomous",
        icon=":material/all_inclusive:",
    ),
    "14": st.Page(
        "demos/agents/research/page.py",
        title="14 Deep research",
        url_path="research",
        icon=":material/travel_explore:",
    ),
    "15": st.Page(
        "demos/agents/sql_analyst/page.py",
        title="15 Text-to-SQL analyst",
        url_path="sql_analyst",
        icon=":material/table_chart:",
    ),
}

# Temporary: Step 0's checks live here, and it goes when the demos that replace
# its checks are built. Not in DEMOS, not in the catalogue.
DEBUG_PAGE = st.Page("_debug.py", title="Core checks", url_path="debug", icon=":material/build:")

# Constrains the reading measure. The app is wide for the demo pages, where the
# trajectory and the trace genuinely use the width; a page of sentences does not.
HOME_WIDTH = 1040
SHOW_ALL = "home_show_all"


def open_demo(index: str) -> None:
    page = BUILT_PAGES.get(index)
    if page is None:
        st.warning(f"Demo {index} — {DEMOS[index][0]} — has no page yet.")
        return
    st.switch_page(page)


def symptom_tile(symptom: str, indices: list[str], key: str) -> None:
    # No `height`: the tile is exactly as tall as its text. Any height at all --
    # a pixel count or "stretch" -- turns the container into a scroll surface.
    with st.container(border=True):
        st.markdown(symptom)
        with st.container(horizontal=True, gap="small"):
            for index in indices:
                if st.button(
                    f"`{index}`  {DEMOS[index][0]}", key=f"{key}_{index}", type="tertiary"
                ):
                    open_demo(index)


def baseline_tile() -> None:
    with st.container(border=True):
        st.markdown(BASELINE_PROMPT)
        if st.button("Open ReAct", key="symptom_baseline", type="primary"):
            open_demo("01")


def catalog_card(index: str) -> None:
    name, hook, shape = DEMOS[index]
    with st.container(border=True):
        if st.button(f"`{index}`  {name}", key=f"catalog_{index}", type="tertiary"):
            open_demo(index)
        st.markdown(hook)
        # The shape is why the catalogue is worth showing: the cards share a width,
        # so the control flows line up and visibly diverge card to card.
        st.caption(f"`{shape}`")


def full_catalog() -> None:
    for family, (note, indices) in FAMILIES.items():
        st.subheader(family, divider="gray")
        st.caption(note)
        st.space("small")
        for row_start in range(0, len(indices), 3):
            cols = st.columns(3, gap="medium")
            for col, index in zip(cols, indices[row_start : row_start + 3]):
                with col:
                    catalog_card(index)
        st.space("small")


def home() -> None:
    with st.container(horizontal_alignment="center"):
        with st.container(width=HOME_WIDTH):
            st.title("GenAI Agentic Lab")
            st.markdown(
                "Sixteen agentic techniques, one runnable demo each. Every one of them exists "
                "because a plain think-act-observe loop broke in a particular way — so start "
                "from the way yours is breaking."
            )
            st.markdown(
                ":gray[Each page gives an agent a task, bounds the run with a step cap, a token "
                "budget and a deadline, and shows every step it took: what it thought, which "
                "tool it called with which arguments, what came back, and what it decided next.]"
            )
            st.divider()

            st.header("What is your agent getting wrong?")
            st.space("small")

            tiles = [
                (lambda s=symptom, i=indices, n=position: symptom_tile(s, i, key=f"symptom_{n}"))
                for position, (symptom, indices) in enumerate(SYMPTOMS)
            ]
            tiles.append(baseline_tile)
            for row_start in range(0, len(tiles), 3):
                cols = st.columns(3, gap="medium")
                for col, render_tile in zip(cols, tiles[row_start : row_start + 3]):
                    with col:
                        render_tile()

            st.space("medium")

            show_all = st.session_state.get(SHOW_ALL, False)
            label = "Hide the full catalogue" if show_all else "Show all sixteen techniques"
            if st.button(label, key="toggle_catalog", width="stretch", icon=":material/list:"):
                st.session_state[SHOW_ALL] = not show_all
                st.rerun()
            if not show_all:
                st.caption(
                    "Every technique, grouped by the problem it solves, with the control flow "
                    "each one runs — useful once you know what you are looking for."
                )
            else:
                st.space("small")
                full_catalog()


HOME_PAGE = st.Page(home, title="Home", icon=":material/home:", default=True)


def sidebar_nav(current: st.Page) -> None:
    """Home link, then the demos inside a collapsed expander.

    Streamlit's own sidebar navigation is hidden and rebuilt here because its
    `expanded=` argument cannot be relied on: the first time a reader clicks
    "View N more", the frontend writes `sidebarNavState=expanded` into
    localStorage and force-expands the list on every page from then on, whatever
    `expanded=` says — and the flag is per origin including port. There is no way
    to clear it from Python, and an expanded list of sixteen pushes the task
    presets, the budget caps and "Clear my data" below the fold.

    The expander is keyed on the current page, so every navigation gives it a
    fresh identity: arriving at a demo always finds the list closed.
    """
    st.sidebar.page_link(HOME_PAGE)
    with st.sidebar.expander("Demos", expanded=False, key=f"nav_demos_{current.title}"):
        if BUILT_PAGES:
            for page in BUILT_PAGES.values():
                st.page_link(page)
        else:
            st.caption("No demos built yet.")
    st.sidebar.page_link(DEBUG_PAGE)


nav = st.navigation(
    {"": [HOME_PAGE], "Demos": list(BUILT_PAGES.values()), "Checks": [DEBUG_PAGE]},
    position="hidden",
)
# Rendered before the page runs, so the page's own sidebar content sits below it.
sidebar_nav(nav)
nav.run()
