# Field Service Bulletin FSB-114 — HX-40 sustained throughput

Halden Instruments Ltd · Issued 2026-03-09 · Supersedes FSB-109 · Classification: Advisory

## Summary

Field data collected from 214 instruments over the twelve months to February
2026 shows that the HX-40 does **not** sustain its rated throughput of 1,200
samples per hour in normal laboratory conditions. The sustained figure, measured
over a full working shift rather than a single hour, is **840 samples per hour**.

This bulletin exists because the discrepancy has been raised as a complaint by
eleven customers since October 2025, and in three cases has been treated as a
fault when it is in fact expected behaviour.

## What is happening

Above an ambient temperature of 26 °C, the HX-40's optical bench enters thermal
derating: the lamp flash rate is reduced to hold the detector within its
calibrated temperature band. Derating is progressive rather than stepped, and it
is not reported to the operator on firmware below 4.3.1 — the instrument simply
runs slower, which is why it is so often read as a fault.

Three factors make derating more likely:

1. **Rear clearance below 150 mm.** The single largest contributor. An
   instrument in a bench recess recirculates its own exhaust.
2. **Carousel reloading during the run.** Each door opening admits room air.
3. **Lamp age.** A lamp past 3 million flashes runs hotter for the same output.

An instrument with adequate clearance, in a 21 °C room, with a lamp under
1 million flashes, will sustain close to its rated figure. Few laboratories meet
all three conditions for a whole shift, which is the gap between the product
brief's number and this one.

## Affected instruments

All HX-40 instruments are affected. The behaviour is by design; it is the
*reporting* of it that changed. Instruments in the serial range `HX40-02100`
through `HX40-04980` shipped with a fan curve that begins ramping at 28 °C
rather than 24 °C, and these derate noticeably earlier than instruments outside
that range. A fan curve update is included in firmware 4.3.1.

## Remedy

1. Update to firmware **4.3.1** or later. This adds a derating indicator to the
   status bar and the result log, and applies the corrected fan curve. It does
   not increase throughput on its own.
2. Verify 150 mm rear clearance and that the exhaust is not directed at a wall
   or another instrument.
3. Where a laboratory's throughput commitment depends on the rated figure,
   plan against **840 samples per hour**, not 1,200.

Field engineers should not replace lamps or optical benches for slow running
alone. Of the 37 warranty lamp replacements raised for slow running in 2025,
31 were later found to be derating rather than lamp failure, at an internal cost
of approximately £64,000.

## Customer communication

Sales and support staff should describe 1,200 samples per hour as a *peak* rate
measured under test method HAL-TM-19, and 840 samples per hour as the *sustained*
rate a laboratory should plan around. Product marketing was asked on 2026-03-02
to reissue the product brief with both figures; at the time of writing the brief
still carries only the peak figure, and revision D remains the current issue.

Questions about this bulletin go to the field engineering lead, not to product
management.
