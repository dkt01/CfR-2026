"""Dependency-free plotting.

matplotlib would be nicer, but it is one more thing that has to be installed on
a machine that may have no network when it matters.  Hand-written SVG opens in
any browser, diffs as text, and embeds in a report without a binary blob.
"""

import html

__all__ = ["scatter"]

_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def _bounds(series):
    xs = [x for entry in series for x in entry["x"]]
    ys = [y for entry in series for y in entry["y"]]
    if not xs or not ys:
        return 0.0, 1.0, 0.0, 1.0
    x_lo, x_hi, y_lo, y_hi = min(xs), max(xs), min(ys), max(ys)
    if x_hi - x_lo < 1e-12:
        x_lo, x_hi = x_lo - 0.5, x_hi + 0.5
    if y_hi - y_lo < 1e-12:
        y_lo, y_hi = y_lo - 0.5, y_hi + 0.5
    pad_y = (y_hi - y_lo) * 0.08
    return x_lo, x_hi, y_lo - pad_y, y_hi + pad_y


def scatter(path, series, title="", x_label="", y_label="", width=780, height=440):
    """Write an SVG with one or more series.

    Each series is {'label': str, 'x': [...], 'y': [...], 'mode': 'line'|'points'}.
    """
    series = [entry for entry in series if entry.get("x") and entry.get("y")]
    if not series:
        return None

    left, right, top, bottom = 68, 18, 40, 52
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_lo, x_hi, y_lo, y_hi = _bounds(series)

    def sx(value):
        return left + (value - x_lo) / (x_hi - x_lo) * plot_w

    def sy(value):
        return top + plot_h - (value - y_lo) / (y_hi - y_lo) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui,sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2:.0f}" y="22" text-anchor="middle" font-size="15" '
        f'font-weight="600">{html.escape(title)}</text>',
    ]

    for step in range(5):
        value = y_lo + (y_hi - y_lo) * step / 4
        y = sy(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'stroke="#e6e6e6"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#555">'
            f"{value:.4g}</text>"
        )
        value = x_lo + (x_hi - x_lo) * step / 4
        x = sx(value)
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 18:.0f}" text-anchor="middle" '
            f'fill="#555">{value:.4g}</text>'
        )

    parts.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" '
        f'fill="none" stroke="#333"/>'
    )

    for index, entry in enumerate(series):
        colour = _PALETTE[index % len(_PALETTE)]
        points = list(zip(entry["x"], entry["y"]))
        if entry.get("mode", "points") == "line":
            path_data = " ".join(
                f"{'M' if position == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
                for position, (x, y) in enumerate(points)
            )
            parts.append(
                f'<path d="{path_data}" fill="none" stroke="{colour}" stroke-width="1.6"/>'
            )
        else:
            for x, y in points:
                parts.append(
                    f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" fill="{colour}"/>'
                )
        if entry.get("label"):
            y_legend = top + 14 + index * 16
            parts.append(
                f'<rect x="{left + 10}" y="{y_legend - 8}" width="10" height="10" fill="{colour}"/>'
            )
            parts.append(
                f'<text x="{left + 26}" y="{y_legend + 1}" fill="#333">'
                f"{html.escape(entry['label'])}</text>"
            )

    parts.append(
        f'<text x="{left + plot_w / 2:.0f}" y="{height - 12}" text-anchor="middle" '
        f'fill="#333">{html.escape(x_label)}</text>'
    )
    parts.append(
        f'<text x="16" y="{top + plot_h / 2:.0f}" text-anchor="middle" fill="#333" '
        f'transform="rotate(-90 16 {top + plot_h / 2:.0f})">{html.escape(y_label)}</text>'
    )
    parts.append("</svg>")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))
    return path
