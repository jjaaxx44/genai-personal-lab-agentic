# On-call runbook — HX-40 support desk

Halden Instruments Ltd · Internal · Revision 11 · Effective 2026-01-06

This runbook covers the first hour of any HX-40 incident. It does not cover
commissioning, and it does not cover the HX-32, which has its own runbook
(HAL-RB-0032).

## The thirty-minute rule

If you have not restored service or identified a root cause within **30 minutes**
of picking up an incident, escalate. Escalating is not a failure; holding an
incident past thirty minutes without escalating is. The rule exists because the
2026-02-14 incident ran for four hours on a first-line desk before anyone with
optical-bench experience saw it.

## Severity

| Severity | Definition | First response | Escalate after |
|---|---|---|---|
| S1 | Instrument down, no workaround, customer cannot run samples | 15 minutes | 30 minutes |
| S2 | Degraded — running, but results delayed or throughput materially reduced | 1 hour | 4 hours |
| S3 | Cosmetic, documentation, or a question | 1 business day | — |
| S4 | Enhancement request | Logged, no target | — |

Slow running alone is **S2, not S1**, and before raising it at all, check
FSB-114: sustained throughput below the rated figure is usually thermal derating
rather than a fault.

## Error codes

| Code | Meaning | First action |
|---|---|---|
| E-207 | Carousel jam | Power down, open the lid, check for a cartridge seated proud of position 1 |
| E-311 | Lamp drift beyond calibration band | Check flash count; over 3.5 million, schedule a lamp change |
| E-402 | Detector temperature out of band | Check ambient and rear clearance before anything else |
| E-518 | Cartridge lot not recognised | Check the lot is within its 14-month shelf life and was received after 2025-04-01 |
| E-771 | Local result buffer full | Instrument has been offline more than 90 days; restore the network link |

E-311 is the code that matters most. It is the only error in this table that can
produce **plausible but wrong results** rather than stopping the run, which is
what made the February 2026 incident as expensive as it was. Any E-311 in the
last 30 days of a customer's log must be treated as a possible result-integrity
issue, not just a maintenance item.

## Paging

- First line: support desk, weekdays 08:00–18:00 UK.
- Second line: field engineering, via the rota in the on-call calendar.
- Optical bench specialists: two people, both in Manchester, both opt-in for
  weekend cover. Do not page them directly for anything other than E-311 or
  E-402.
- Anything that may have produced incorrect customer results goes to the quality
  lead **immediately**, in parallel with technical work, not after it.

## What not to do

Do not replace a lamp to resolve slow running. Do not advise a customer to raise
their ambient temperature set point to reduce condensation, which has been
suggested twice and makes derating worse. Do not close an S1 on the customer's
word alone; confirm from the instrument log that the last 20 runs completed.
