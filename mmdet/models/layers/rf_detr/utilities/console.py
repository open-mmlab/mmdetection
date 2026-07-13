# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Rich console helper for the training callback stack."""

from __future__ import annotations

from typing import Any

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.table import Table
except ImportError:
    Console = None  # type: ignore[assignment, misc]
    Group = None  # type: ignore[assignment, misc]
    Live = None  # type: ignore[assignment, misc]
    Table = None  # type: ignore[assignment, misc]
    _IS_RICH_AVAILABLE = False
else:
    _IS_RICH_AVAILABLE = True


def _get_rich_console(trainer: Any) -> Any:
    """Return a Rich Console appropriate for the current training context.

    When a ``RichProgressBar`` callback is active, its ``_console`` is the Rich global
    singleton that owns the Live progress display.  Printing through that same console
    instance ensures output appears correctly above the progress bars rather than
    conflicting with the Live display's cursor positioning (particularly on Windows).

    Falls back to ``Console(force_terminal=True)`` for plain-terminal and TQDM contexts.

    Args:
        trainer: The PTL Trainer (or any object with a ``callbacks`` attribute).

    Returns:
        A Rich ``Console`` instance suitable for printing metric tables.

    Note:
        Reads ``RichProgressBar._console``, a private PTL attribute. When ``_console`` is
        ``None`` (outside an active fit/validate/test stage) or the attribute is renamed in a
        future PTL release, the function silently falls back to ``Console(force_terminal=True)``
        — a fresh console with an empty live stack that may reintroduce cursor conflicts on
        Windows. Subclasses and theme wrappers of ``RichProgressBar`` are matched via MRO.
    """
    if Console is None:
        raise RuntimeError(  # pragma: no cover
            "_get_rich_console called when Rich is not installed; check _IS_RICH_AVAILABLE first."
        )

    for cb in getattr(trainer, "callbacks", []):
        if any(cls.__name__ == "RichProgressBar" for cls in type(cb).__mro__):
            cb_console = getattr(cb, "_console", None)
            if cb_console is not None:
                return cb_console
    return Console(force_terminal=True)


def _has_progress_bar(trainer: Any) -> bool:
    """Return whether the trainer has a Lightning progress bar callback.

    Args:
        trainer: The PTL Trainer (or any object with a ``callbacks`` attribute).

    Returns:
        ``True`` when any callback class name ends with ``"ProgressBar"``.
    """
    callbacks = getattr(trainer, "callbacks", [])
    return any(callback.__class__.__name__.endswith("ProgressBar") for callback in callbacks)


def _render_overall_merged(title_pfx: str, overall: dict[str, float], max_dets: int) -> str:
    """Render the overall metrics table with merged group-header cells.

    Uses only plain Unicode box-drawing characters (no ANSI colour codes) so the output renders correctly in both
    terminals and Jupyter/Colab notebook widgets.

    Args:
        title_pfx: Capitalised split name used in the title (e.g. ``"Val"``).
        overall: Ordered mapping of metric label → scalar value.
        max_dets: Maximum detection count used for the mAR column label.

    Returns:
        Multi-line plain-text string ready to pass to ``console.print()``.
    """

    def _fmt(v: float) -> str:
        if v != v or v < 0:  # NaN or pycocotools sentinel -1 → em-dash
            return "—"
        return f"{v:.4f}"

    mar_lbl = f"@{max_dets}"
    mar_key = f"mAR @{max_dets}"

    groups: list[tuple[str, list[tuple[str, str]]]] = [
        (
            "mAP",
            [
                ("50:95", _fmt(overall["mAP 50:95"])),
                ("50", _fmt(overall["mAP 50"])),
                ("75", _fmt(overall["mAP 75"])),
            ],
        ),
        ("mAR", [(mar_lbl, _fmt(overall[mar_key]))]),
        (
            "F1 sweep",
            [
                ("F1", _fmt(overall["F1"])),
                ("Prec", _fmt(overall["Precision"])),
                ("Recall", _fmt(overall["Recall"])),
            ],
        ),
    ]
    if "segm mAP 50:95" in overall:
        groups.append(
            (
                "segm mAP",
                [
                    ("50:95", _fmt(overall["segm mAP 50:95"])),
                    ("50", _fmt(overall["segm mAP 50"])),
                ],
            )
        )

    flat: list[tuple[str, str]] = [(s, v) for _, cols in groups for s, v in cols]
    widths: list[int] = [max(len(s), len(v)) + 2 for s, v in flat]

    col = 0
    for grp, cols in groups:
        nc = len(cols)
        cell_w = sum(widths[col : col + nc]) + (nc - 1)
        needed = len(grp) + 2
        if needed > cell_w:
            for k in range(needed - cell_w):
                widths[col + k % nc] += 1
        col += nc

    spans: list[tuple[int, int, str]] = []
    col = 0
    for grp, cols in groups:
        nc = len(cols)
        spans.append((col, col + nc - 1, grp))
        col += nc

    grp_ends = {end for start, end, _ in spans[:-1]}
    n = len(flat)

    def grp_w(start: int, end: int) -> int:
        """Return merged cell width for columns start..end inclusive."""
        return sum(widths[start : end + 1]) + (end - start)

    heavy_horizontal = "━"
    light_horizontal = "─"
    heavy_vertical = "┃"
    light_vertical = "│"
    top_left_corner, top_right_corner = "┏", "┓"
    top_t_down = "┳"
    transition_left, transition_right = "┡", "┩"
    group_join = "╇"
    subgroup_join = "┯"
    mid_left, mid_right, mid_cross = "├", "┤", "┼"
    bottom_left_corner, bottom_right_corner, bottom_t_up = "└", "┘", "┴"

    inner_w = sum(widths) + n - 1
    title = f"{title_pfx} — Overall Metrics"
    title_line = title.center(inner_w + 2)

    r1 = top_left_corner
    for i, (s, e, _) in enumerate(spans):
        r1 += heavy_horizontal * grp_w(s, e)
        r1 += top_t_down if i < len(spans) - 1 else top_right_corner

    r2 = heavy_vertical
    for s, e, grp in spans:
        r2 += grp.center(grp_w(s, e)) + heavy_vertical

    r3 = transition_left
    for i, w in enumerate(widths):
        r3 += heavy_horizontal * w
        if i < n - 1:
            r3 += group_join if i in grp_ends else subgroup_join
    r3 += transition_right

    r4 = light_vertical
    for i, (sub, _) in enumerate(flat):
        r4 += sub.center(widths[i]) + light_vertical

    r5 = mid_left
    for i, w in enumerate(widths):
        r5 += light_horizontal * w
        r5 += mid_cross if i < n - 1 else mid_right

    r6 = light_vertical
    for i, (_, val) in enumerate(flat):
        r6 += val.center(widths[i]) + light_vertical

    r7 = bottom_left_corner
    for i, w in enumerate(widths):
        r7 += light_horizontal * w
        r7 += bottom_t_up if i < n - 1 else bottom_right_corner

    return "\n".join([title_line, r1, r2, r3, r4, r5, r6, r7])


def _build_summary_renderable(
    title_pfx: str,
    overall_rendered: str,
    per_class: list[dict[str, Any]],
) -> Group:
    """Build a single Rich renderable for overall and per-class metric tables.

    Args:
        title_pfx: Split label (e.g. ``"Val"`` or ``"Test"``).
        overall_rendered: Pre-rendered overall table string.
        per_class: Per-class dicts with keys ``name``, ``ap``, ``ar``,
            ``f1``, ``precision``, ``recall``; skipped when empty.

    Returns:
        Rich ``Group`` renderable containing the overall table and, when present,
        the per-class table.
    """
    assert Group is not None

    def _fmt(v: float) -> str:
        if v != v or v < 0:  # NaN or pycocotools sentinel -1 → em-dash
            return "—"
        return f"{v:.4f}"

    renderables: list[Any] = [overall_rendered]
    if per_class:
        assert Table is not None

        table = Table(
            title=f"{title_pfx} — Per-class Metrics",
            title_style="bold cyan",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Class", style="dim", no_wrap=True)
        table.add_column("AP 50:95", justify="right")
        table.add_column("AR", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        for row in per_class:
            table.add_row(
                row["name"],
                _fmt(row["ap"]),
                _fmt(row["ar"]),
                _fmt(row["f1"]),
                _fmt(row["precision"]),
                _fmt(row["recall"]),
            )
        renderables.append(table)
    return Group(*renderables)


def _render_summary_tables(
    console: Any,
    title_pfx: str,
    overall_rendered: str,
    per_class: list[dict[str, Any]],
) -> None:
    """Print overall and per-class metric tables to ``console``.

    Args:
        console: Rich ``Console`` instance to print to.
        title_pfx: Split label (e.g. ``"Val"`` or ``"Test"``).
        overall_rendered: Pre-rendered overall table string.
        per_class: Per-class dicts with keys ``name``, ``ap``, ``ar``,
            ``f1``, ``precision``, ``recall``; skipped when empty.
    """
    if not _IS_RICH_AVAILABLE:
        return
    console.print(_build_summary_renderable(title_pfx, overall_rendered, per_class))
