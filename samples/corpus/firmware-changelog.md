# HX-40 firmware changelog

Halden Instruments Ltd · Maintained by the firmware team · Last updated 2026-05-14

Versions are listed newest first. A release marked *service visit* cannot be
applied over the network and needs an engineer on site.

## 4.4.0 — planned, not released

Target 2026-05-29. Slipped once from 2026-04-24.

- Service reminders driven by lamp flash count as well as elapsed months
  (post-mortem action 1). A lamp past 3.5 million flashes raises a maintenance
  notice; past 4 million, a warning that cannot be dismissed for more than
  seven days.
- Result log records the derating state alongside each result, so a slow run can
  be distinguished from a fault after the fact.
- Known issue carried forward: the status bar derating indicator added in 4.3.1
  does not appear on the remote web view, only on the instrument panel.

## 4.3.1 — released 2026-03-06

- Derating indicator in the status bar and the result log (FSB-114).
- Corrected fan curve for instruments in serial range `HX40-02100` to
  `HX40-04980`: ramp now begins at 24 °C rather than 28 °C.
- Fixed a fault where E-771 was raised at 85 days offline rather than 90.

## 4.3.0 — released 2026-01-19

- Cartridge lot validation against the 14-month shelf life (E-518).
- Network buffer increased from 7 days to 14 days for connected instruments.
- Security: TLS 1.2 is now the minimum for the laboratory information system
  link. Instruments talking to an older LIS will lose the link on update; this
  caused eleven support calls in the week after release and should have been
  called out in the release note, which it was not.

## 4.2.0 — released 2025-10-28

Shipping version at HX-40 launch.

- First release with six-channel support.
- Carousel jam detection (E-207) rewritten after false positives in beta.

## 4.1.4 — released 2025-08-12 — *service visit*

- Last release for instruments below serial `HX40-02100`. Instruments on 4.1.x
  cannot be updated over the network and need a service visit to reach 4.3.x.
- Detector temperature band widened by 0.5 °C at both ends, to reduce nuisance
  E-402 alerts in laboratories without air conditioning.

## Support policy

The current release and the one before it are supported. When 4.4.0 ships, 4.3.0
leaves support and 4.3.1 becomes the oldest supported version. Instruments on
4.1.x are out of support for defects but will continue to receive security fixes
until 2027-08-12.
