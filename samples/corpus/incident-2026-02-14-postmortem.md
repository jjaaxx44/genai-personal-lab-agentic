# Post-mortem: voided results at Marbury Clinical Labs, 2026-02-14

Halden Instruments Ltd · Internal · Author: quality lead · Reviewed 2026-02-28

## What happened

Between 2026-01-27 and 2026-02-14, instrument `HX40-03911` at Marbury Clinical
Labs produced results that drifted progressively low across all six optical
channels. The drift was not detected by the instrument, by the laboratory's
internal controls, or by Halden's support desk. On 2026-02-14 a Marbury
scientist noticed that a control sample had trended in one direction for eleven
consecutive days and stopped the instrument.

**2,847 results across 19 days were voided and re-run.** Marbury's own estimate
of the cost to them, including overtime and reagents, was £38,000. Halden has
credited that amount and is not disputing it.

## Root cause

Three things had to line up, and all three did.

1. **The lamp was at 3.9 million flashes**, past its 4-million rated life by the
   only measure anyone was tracking, but the instrument's service reminder is
   driven by elapsed months rather than flash count. The lamp was 14 months old,
   so no reminder had fired.

2. **The E-311 alert threshold had been widened.** During commissioning in
   2024-09, an engineer widened the lamp drift band on this instrument to
   suppress what were believed to be nuisance alerts. The change was recorded in
   the engineer's visit notes but not in the instrument's configuration audit,
   and no process reviewed it afterwards. The widened band was approximately
   twice the factory value.

3. **The drift was monotonic and slow.** At roughly 0.4 % per day it never
   tripped a day-on-day check, only a trend check, and Marbury's control review
   was weekly against limits rather than against trend.

The single-point failure was the second one. With a factory-width band, the
instrument would have raised E-311 on or around 2026-02-01, thirteen days before
the customer caught it.

## What made it worse

The incident sat on the first-line desk for four hours before an optical bench
specialist saw it, because the ticket was opened as "results look odd" and
triaged S3. There was no rule at the time that a possible result-integrity issue
goes to the quality lead immediately. That rule now exists and is in revision 11
of the on-call runbook.

## Actions

| # | Action | Owner | Due | Status |
|---|---|---|---|---|
| 1 | Service reminders driven by flash count as well as elapsed months | Firmware | 2026-05-29 | In 4.4.0, not yet released |
| 2 | Configuration audit records every threshold change, with the engineer's reason | Service systems | 2026-04-30 | Done |
| 3 | Any E-311 in the last 30 days triggers a result-integrity review | Support process | 2026-03-02 | Done, runbook rev 11 |
| 4 | Audit every instrument commissioned before 2025-01-01 for widened thresholds | Field engineering | 2026-06-30 | 61 of 148 audited |
| 5 | Customer-facing guidance on trend-based control review | Applications | 2026-07-31 | Not started |

Action 4 is the one that still carries risk: 87 instruments remain unaudited,
and the widened-threshold pattern has already been found on four of the 61
audited so far.

## What we are not doing

We are not recalling instruments. The defect is a configuration and process
failure, not a hardware fault, and a recall would not address either. This was
argued at the review on 2026-02-24 and the decision was recorded there.
