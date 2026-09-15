#!/usr/bin/env python3
"""Fold a run's fitted values into config/vehicle.yaml, with provenance.

    ./scripts/apply_vehicle_patch.py ~/cfr_runs/20260915T141203Z_coastdown

Edits the file as TEXT rather than loading and re-dumping it.  vehicle.yaml is
mostly comments explaining where each number came from and why it matters, and
PyYAML would throw every one of them away on the first round trip - which would
quietly convert the most useful file in the campaign into a list of bare
numbers.

Each value it touches gets `provenance: measured`, the run directory that
justifies it, and the date.  A number in that file should always be able to
answer "says who?".
"""

import argparse
import os
import re
import sys
from datetime import datetime, timezone

import yaml

DEFAULT_VEHICLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cfr_arduino_bridge",
    "config",
    "vehicle.yaml",
)


class PatchError(RuntimeError):
    pass


def _format_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # Enough digits to round-trip a fit without printing noise.
        return f"{value:.6g}"
    return str(value)


def _find_entry(lines, section, key):
    """Return (start, end) line indices of `section: -> key:`'s block body."""
    section_pattern = re.compile(rf"^{re.escape(section)}:\s*$")
    key_pattern = re.compile(rf"^  {re.escape(key)}:\s*$")

    index = next(
        (i for i, line in enumerate(lines) if section_pattern.match(line)), None
    )
    if index is None:
        raise PatchError(f'no section "{section}" in vehicle.yaml')

    index += 1
    while index < len(lines) and not re.match(r"^\S", lines[index]):
        if key_pattern.match(lines[index]):
            start = index + 1
            end = start
            while end < len(lines) and (
                lines[end].startswith("    ") or not lines[end].strip()
            ):
                end += 1
            return start, end
        index += 1
    raise PatchError(f'no key "{key}" under "{section}" in vehicle.yaml')


def _rewrite_block(block, value, run_dir, today):
    """Replace value/provenance inside one entry, leaving its comments alone."""
    if isinstance(value, list):
        anchor = "rows:"
        rendered = ["    rows:"] + [
            "      - [" + ", ".join(_format_scalar(item) for item in row) + "]"
            for row in value
        ]
    else:
        anchor = "value:"
        rendered = [f"    value: {_format_scalar(value)}"]

    out, rendered_start, seen_provenance = [], None, False
    skipping_rows = False
    for line in block:
        stripped = line.strip()
        if skipping_rows:
            # Drop the old table body, which follows `rows:` as indented items.
            if stripped.startswith("- ") or not stripped:
                continue
            skipping_rows = False
        if stripped.startswith(anchor) and rendered_start is None:
            rendered_start = len(out)
            out.extend(rendered)
            skipping_rows = isinstance(value, list)
            continue
        if stripped.startswith("provenance:"):
            out.append("    provenance: measured")
            seen_provenance = True
            continue
        if stripped.startswith(("run:", "measured_utc:")):
            continue  # re-emitted below, so re-applying never stacks them up
        out.append(line)

    if rendered_start is None:
        raise PatchError(f'entry has no "{anchor}" line to replace')

    # Directly after the rendered value.  Anywhere else and the audit lines land
    # inside a rows list, which silently truncates the table at that point.
    insert_at = rendered_start + len(rendered)
    audit = [f"    run: {run_dir}", f"    measured_utc: '{today}'"]
    if not seen_provenance:
        audit.insert(0, "    provenance: measured")
    out[insert_at:insert_at] = audit
    return out


def apply_patch(vehicle_path, values, run_dir, today=None):
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with open(vehicle_path, "r", encoding="utf-8") as handle:
        lines = handle.read().split("\n")

    applied, skipped = [], []
    for dotted, value in sorted(values.items()):
        if "." not in dotted:
            skipped.append((dotted, "not a section.key path"))
            continue
        section, _, key = dotted.partition(".")
        try:
            start, end = _find_entry(lines, section, key)
            lines[start:end] = _rewrite_block(lines[start:end], value, run_dir, today)
            applied.append(dotted)
        except PatchError as error:
            # A fit can produce a diagnostic value with no home in vehicle.yaml
            # (a cross-check ratio, a scoring metric).  That is not an error; it
            # belongs in the run's report, not in the vehicle description.
            skipped.append((dotted, str(error)))

    with open(vehicle_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return applied, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", help="run directory containing vehicle_patch.yaml")
    parser.add_argument(
        "--vehicle", default=DEFAULT_VEHICLE, help="path to vehicle.yaml"
    )
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args(argv)

    patch_path = os.path.join(os.path.expanduser(args.run_dir), "vehicle_patch.yaml")
    if not os.path.isfile(patch_path):
        print(
            f"error: no vehicle_patch.yaml in {args.run_dir} - run analyze_run.py first",
            file=sys.stderr,
        )
        return 1
    with open(patch_path, "r", encoding="utf-8") as handle:
        patch = yaml.safe_load(handle) or {}
    values = patch.get("values") or {}
    if not values:
        print("nothing to apply")
        return 0

    target = args.vehicle
    if args.dry_run:
        import shutil
        import tempfile

        target = os.path.join(tempfile.mkdtemp(), "vehicle.yaml")
        shutil.copy(args.vehicle, target)

    applied, skipped = apply_patch(target, values, patch.get("run", args.run_dir))

    for name in applied:
        print(f"  updated  {name} = {values[name]}")
    for name, reason in skipped:
        print(f"  skipped  {name} ({reason})")
    if args.dry_run:
        print(f"\ndry run - wrote to {target}, left {args.vehicle} alone")
    elif applied:
        print(f"\n{len(applied)} value(s) updated in {args.vehicle}")
        print("Review the diff, then rebuild so the SDF picks the changes up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
