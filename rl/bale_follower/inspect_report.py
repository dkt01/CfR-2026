#!/usr/bin/env python3
"""Print one region's rows from a corridor validation report.

Separate from validate_corridor.py because the summary there answers "is it
good enough", and when the answer is no the next question is always "wrong
in which direction", which needs the individual poses rather than a mean of
absolute errors. A mean absolute error cannot distinguish a systematic bias
-- which is a sign or scale bug and usually fixable -- from scatter, which
means the geometry is not being seen at all.

    python3 inspect_report.py corridor_report.json helical_ramp
    python3 inspect_report.py corridor_report.json            # region tally
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rows = json.loads(Path(sys.argv[1]).read_text())

    if len(sys.argv) < 3:
        tally: dict[str, int] = {}
        confident: dict[str, int] = {}
        for row in rows:
            tally[row["region"]] = tally.get(row["region"], 0) + 1
            if row["confidence"] >= 0.5:
                confident[row["region"]] = confident.get(row["region"], 0) + 1
        print(f"{'region':<20} {'poses':>6} {'confident':>10}")
        for region in sorted(tally):
            print(f"{region:<20} {tally[region]:>6} {confident.get(region, 0):>10}")
        return 0

    region = sys.argv[2]
    selected = [row for row in rows if row["region"] == region]
    if not selected:
        print(
            f"no rows for {region!r}; regions present: "
            f"{sorted({row['region'] for row in rows})}"
        )
        return 1

    print(
        f"{'s':>6} {'conf':>5} {'true_off':>9} {'off':>7} {'d_off':>7} "
        f"{'true_hdg':>9} {'hdg':>7} {'d_hdg':>7} {'width':>6} {'scan_min':>8}"
    )
    for row in sorted(selected, key=lambda r: (r["s"], r["true_offset"])):
        print(
            f"{row['s']:>6.1f} {row['confidence']:>5.2f} "
            f"{row['true_offset']:>9.2f} {row['offset']:>7.3f} "
            f"{row['offset_error']:>7.3f} "
            f"{row['true_heading_deg']:>9.1f} {row['heading_deg']:>7.1f} "
            f"{row['heading_error_deg']:>7.1f} "
            f"{row['half_width']:>6.2f} {row['scan_min']:>8.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
