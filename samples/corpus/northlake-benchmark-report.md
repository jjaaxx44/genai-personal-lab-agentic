# Independent benchmark: HX-40 throughput under varying ambient temperature

Northlake Analytical Services · Report NL-2026-018 · Commissioned by Halden Instruments · 2026-04-22

## Scope

Halden Instruments commissioned Northlake to measure HX-40 throughput
independently, following the disagreement between the HX-40 product brief
(HAL-PB-0040 rev D, 1,200 samples per hour) and Field Service Bulletin FSB-114
(840 samples per hour). Northlake was asked to establish which figure holds,
under what conditions, and whether the two can be reconciled.

Two instruments were tested: `HX40-03442`, which falls inside the fan-curve
serial range named in FSB-114, and `HX40-05107`, which falls outside it. Both
ran firmware 4.3.1. Both had lamps at approximately 900,000 flashes. Rear
clearance was set to 150 mm for all runs, which is better than most of the field
population.

## Method

Each instrument ran a single assay type continuously for six hours at each of
four ambient temperatures, with the carousel reloaded on a fixed 20-minute
cadence. Throughput was recorded per hour and averaged across the last five
hours of each run, discarding the first hour as warm-up.

## Results

| Ambient | HX40-03442 (in range) | HX40-05107 (out of range) |
|---|---|---|
| 18 °C | 1,162 samples/h | 1,171 samples/h |
| 21 °C | 1,150 samples/h | 1,158 samples/h |
| 24 °C | 1,034 samples/h | 1,120 samples/h |
| 28 °C | 905 samples/h | 962 samples/h |

## Findings

1. **Neither published figure is wrong, and neither is sufficient alone.** The
   product brief's 1,200 samples per hour is reachable only as a peak: our best
   sustained hour on any instrument was 1,183, and our best five-hour average
   was 1,171, both at 18 °C. FSB-114's 840 is conservative for a well-installed
   instrument but realistic for the field population, which includes instruments
   with inadequate clearance and older lamps — neither of which we tested.

2. **The serial range in FSB-114 is real and measurable.** At 24 °C the in-range
   instrument was 7.7 % slower than the out-of-range one; at 28 °C, 5.9 %. Below
   24 °C the difference is within measurement noise.

3. **Derating begins earlier than FSB-114 implies.** FSB-114 gives 26 °C as the
   onset. We observed measurable derating from 23.5 °C on the in-range
   instrument, and from approximately 25 °C on the out-of-range one. We
   recommend Halden restate the onset as a range rather than a single figure.

4. **Reloading cadence matters more than the bulletin suggests.** A supplementary
   run at 21 °C with reloading every 5 minutes rather than every 20 gave 1,061
   samples per hour, a 7.7 % penalty at constant ambient temperature.

## Recommendation to the commissioning party

Publish both numbers with their conditions attached, in the same table: a peak
of approximately 1,150–1,170 samples per hour at 21 °C or below, and a planning
figure of 840–950 samples per hour depending on installation quality and ambient
temperature. Publishing one number without its conditions is what produced the
eleven complaints FSB-114 describes.

Northlake has no financial interest in the HX-40 product line beyond the fee for
this report, which was fixed before the results were known.
