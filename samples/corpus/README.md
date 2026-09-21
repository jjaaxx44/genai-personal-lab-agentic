# Corpus

What the `search_corpus` tool reads. Drop `.md` or `.txt` files in here;
`core/corpus.py` chunks and indexes them on first use, and re-indexes a file when
its hash changes.

This is deliberately not a PDF pipeline — that is the sibling RAG lab's subject,
and keeping a PDF library out of this repo is also what keeps its dependencies
permissively licensed (see `LICENSE`).

## What is in here now

A synthetic corpus about a fictional company, **Halden Instruments**, and its
HX-40 benchtop analyser. Seven documents, written to give the agents something to
work at rather than something to look up:

| File | What it is |
|---|---|
| `hx40-product-brief.md` | The product specification, including the rated throughput figure |
| `fsb-114-thermal-derating.md` | A field service bulletin that **contradicts** the brief's throughput figure |
| `northlake-benchmark-report.md` | An independent benchmark that explains how both figures can be true, and disagrees with the bulletin on a detail |
| `oncall-runbook.md` | Severities, error codes and escalation rules |
| `incident-2026-02-14-postmortem.md` | An incident with a root cause, costs, and an action list with owners and due dates |
| `firmware-changelog.md` | Versions, dates and what each release changed — including the fix the bulletin asks for |
| `procurement-policy.md` | Approval thresholds, lead times and a single-sourcing risk |

Three properties were designed in on purpose, because the demos need them:

- **The answer is rarely in the first chunk.** Most questions worth asking need
  two passages from different files.
- **Facts that appear exactly once** — a part number, a serial range, an error
  code, a due date — so a search that misses is visible as a wrong answer rather
  than a vaguer one.
- **A genuine disagreement.** The brief says 1,200 samples per hour, the bulletin
  says 840, and the benchmark report shows both are measuring different things.
  An agent that reports one number without its conditions has got it wrong, and
  the research and evaluation demos are built on exactly that.

This README is indexed along with everything else, which is harmless but means an
empty corpus would still return a hit.

Everything in these documents is invented. Any resemblance to a real instrument,
company or incident is accidental.

**Replaceable.** This corpus is scaffolding, not content the app depends on.
Delete it and drop your own documents in; the index follows the directory. If
you do, keep the three properties above — most of the demos get dull against a
corpus that answers everything from its first paragraph.
