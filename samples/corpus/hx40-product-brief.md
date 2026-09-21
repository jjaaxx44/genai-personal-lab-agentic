# HX-40 Benchtop Analyser — Product Brief

Halden Instruments Ltd · Document HAL-PB-0040 · Revision D · Issued 2025-11-03

## Overview

The HX-40 is a benchtop optical analyser for routine sample screening in
mid-volume laboratories. It replaces the HX-32, which reached end of sale on
2025-06-30 and leaves support on 2028-06-30. The HX-40 shares the HX-32's
consumable cartridges but not its lamp assembly, which is a common source of
confusion when ordering spares.

The instrument is aimed at laboratories running between 2,000 and 8,000 samples
a week. Below that volume the HX-20 is more economical; above it, customers are
directed to the rack-mounted HR series.

## Headline specification

| Property | Value |
|---|---|
| Rated throughput | 1,200 samples per hour |
| Carousel capacity | 40 positions |
| Sample volume | 20–250 µL |
| Optical channels | 6 |
| Lamp | Xenon flash, part number HAL-LMP-4419 |
| Lamp rated life | 18 months or 4 million flashes, whichever comes first |
| Dimensions | 610 × 480 × 390 mm |
| Mass | 41 kg |
| Power | 230 V AC, 50 Hz, 620 W peak |
| Ambient operating range | 15 °C to 30 °C |
| Firmware at release | 4.2.0 |
| Warranty | 24 months, parts and labour, from date of commissioning |

The rated throughput of **1,200 samples per hour** is measured under the
conditions set out in Halden's internal test method HAL-TM-19: a 21 °C ambient,
a single assay type, no carousel reloading during the measured hour, and a lamp
with fewer than 500,000 flashes on it. The figure is a ceiling, not a service
level, and the brief has been criticised internally for not saying so on the
same page as the table.

## Consumables and part numbers

- Cartridge, standard, box of 500 — HAL-CTG-1100
- Cartridge, high-sensitivity, box of 200 — HAL-CTG-1180
- Lamp assembly — HAL-LMP-4419 (the HX-32 lamp, HAL-LMP-3310, does not fit)
- Carousel, 40-position, spare — HAL-CAR-4040
- Waste bottle and cap kit — HAL-WST-0090

Cartridges carry a 14-month shelf life from manufacture and must be stored
between 4 °C and 25 °C. Cartridges stored above 25 °C for more than 72 hours are
out of specification, and the instrument has no way of detecting this; the only
control is the receiving check at goods-in.

## Installation notes

Commissioning is performed by a Halden engineer and takes approximately four
hours, including the first calibration. The instrument needs 150 mm of clearance
at the rear for airflow. Customers frequently install the HX-40 in a bench recess
that meets the depth requirement but not the clearance one, which is the single
most common cause of the thermal behaviour described in Field Service Bulletin
FSB-114.

Network connectivity is optional. Instruments without a network connection log
locally and hold 90 days of results; instruments with a connection stream to the
laboratory information system and hold 14 days locally as a buffer.

## Support

Standard support covers weekdays 08:00–18:00 UK time, with a four-hour response
target for instruments down and a next-business-day target for everything else.
Extended cover (24/7, two-hour response) is available as contract option EXT-2.

Serial numbers are of the form `HX40-` followed by five digits. Instruments
below `HX40-02100` shipped with firmware 4.1.x and require a service visit to
reach 4.3.x; later instruments can be updated over the network.
