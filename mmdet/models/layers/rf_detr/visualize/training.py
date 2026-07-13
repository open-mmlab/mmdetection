# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Post-training metrics plotting utilities.

Reads the ``metrics.csv`` written by PTL's ``CSVLogger`` (always present after a ``build_trainer``-based run) and builds
seaborn figures grouped by metric type (Loss, AP@0.50, AP@0.50:0.95, AR).

Loss panel shows aggregate and component loss scalars. AP/AR panels show all ``train/``, ``val/``, and ``test/``
columns for each group — both the base and EMA series when EMA is enabled, so both are visible in the legend.

Usage::

    from mmdet.models.layers.rf_detr.visualize.training import plot_metrics
    fig = plot_metrics("output/rfdetr_base/metrics.csv")
    plt.show(fig)
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_AUXILIARY_LOSS_SUFFIX_RE = re.compile(r"_\d+$")
_LEGEND_COLUMNS = 4


def _ensure_headless_backend(matplotlib: Any) -> None:
    """Switch matplotlib to the non-interactive ``Agg`` backend only when safe.

    Forcing ``Agg`` unconditionally would break inline rendering in notebooks, so the switch is skipped when a display
    is available or matplotlib is already in interactive mode.
    """
    if matplotlib.get_backend().lower() == "agg":
        return
    if os.environ.get("DISPLAY") or matplotlib.is_interactive():
        return
    matplotlib.use("Agg")


try:
    import seaborn  # noqa: F401

    _IS_SEABORN_AVAILABLE: bool = True
except ImportError:
    _IS_SEABORN_AVAILABLE = False


def _place_legend_below_axes(ax: Any, *, n_columns: int = _LEGEND_COLUMNS) -> None:
    """Place a compact multi-column legend below a matplotlib axes."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(n_columns, max(1, len(labels))),
        fontsize=9,
        frameon=True,
    )


def _split_metric_column(column: str) -> tuple[str, str]:
    """Split a CSVLogger metric column into split prefix and metric name."""
    split, separator, metric_name = column.partition("/")
    if separator and split in {"train", "val", "test"}:
        return split, metric_name
    return "", column


def _line_style_for_split(split: str) -> str:
    """Return the plotting line style for a metric split."""
    if split == "train":
        return ":"
    if split == "test":
        return "-."
    return "-"


def _plot_columns_on_axes(ax: Any, raw_df: Any, epoch_df: Any, metric_columns: list[str]) -> None:
    """Plot columns with color by metric name and line style by split.

    When seaborn is installed, draws mean ± 1 std-dev bands computed from within-epoch step-level rows in ``raw_df``.
    Falls back to epoch-averaged lines from ``epoch_df`` when seaborn is absent.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for training metric plots. Install it with: pip install matplotlib"
        ) from exc

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    metric_colors: dict[str, str] = {}

    for column in metric_columns:
        split, metric_name = _split_metric_column(column)
        if metric_name not in metric_colors:
            metric_colors[metric_name] = color_cycle[len(metric_colors) % len(color_cycle)] if color_cycle else "C0"
        color = metric_colors[metric_name]
        linestyle = _line_style_for_split(split)

        if _IS_SEABORN_AVAILABLE:
            import seaborn as sns

            col_data = raw_df[["epoch", column]].dropna(subset=[column])
            if not col_data.empty:
                try:
                    sns.lineplot(
                        data=col_data,
                        x="epoch",
                        y=column,
                        ax=ax,
                        errorbar=("sd", 1),
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.7,
                        label=column,
                    )
                except TypeError:
                    # seaborn <0.12: errorbar= kwarg not supported
                    sns.lineplot(
                        data=col_data,
                        x="epoch",
                        y=column,
                        ax=ax,
                        ci="sd",
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.7,
                        label=column,
                    )
        else:
            ax.plot(
                epoch_df["epoch"],
                epoch_df[column],
                linewidth=1.7,
                linestyle=linestyle,
                color=color,
                label=column,
            )


def _build_metric_groups(df: Any) -> dict[str, list[str]]:
    """Build plot groups from numeric PTL CSVLogger metrics.

    Args:
        df: DataFrame-like object with metric columns.

    Returns:
        Non-empty metric groups keyed by subplot title.

    Raises:
        AttributeError: If ``df`` does not provide DataFrame-like columns.
    """

    def _split_cols(*patterns: str) -> list[str]:
        """Return split-prefixed columns whose name contains any of the given patterns."""
        prefixes = ("train/", "val/", "test/")
        return [
            c for c in df.columns if c.startswith(prefixes) and any(p in c for p in patterns) and df[c].notna().any()
        ]

    def _is_loss_col(name: str) -> bool:
        """Return whether a metric column is a loss scalar."""
        if name in {"epoch", "step"}:
            return False
        leaf = name.rsplit("/", maxsplit=1)[-1].lower()
        if _AUXILIARY_LOSS_SUFFIX_RE.search(leaf):
            return False
        return "loss" in leaf or leaf.startswith("kp_")

    loss_cols = [name for name in df.columns if _is_loss_col(str(name)) and df[name].notna().any()]
    detection_map_50 = [c for c in _split_cols("mAP_50", "ema_mAP_50") if "mAP_50_95" not in c and "mAP_75" not in c]
    detection_map_75 = _split_cols("mAP_75", "ema_mAP_75")
    detection_map_50_95 = [
        c for c in _split_cols("mAP_50_95", "ema_mAP_50_95", "/AP/") if "mAP_50" not in c or "mAP_50_95" in c
    ]
    detection_mar = [c for c in _split_cols("mAR", "ema_mAR") if "keypoint_" not in c]
    keypoint_map_50 = [
        c for c in _split_cols("keypoint_map_50", "ema_keypoint_map_50") if "map_50_95" not in c and "map_75" not in c
    ]
    keypoint_map_75 = _split_cols("keypoint_map_75", "ema_keypoint_map_75")
    keypoint_map_50_95 = _split_cols("keypoint_map_50_95", "ema_keypoint_map_50_95")
    keypoint_mar = _split_cols("keypoint_mAR", "ema_keypoint_mAR")
    f1_precision_recall = _split_cols("F1", "precision", "recall")

    metric_groups: dict[str, list[str]] = {
        "Loss": loss_cols,
        "Detection AP@0.50": detection_map_50,
        "Detection AP@0.50:0.95": detection_map_50_95,
        "Detection AP@0.75": detection_map_75,
        "Detection AR": detection_mar,
        "Keypoint AP@0.50": keypoint_map_50,
        "Keypoint AP@0.50:0.95": keypoint_map_50_95,
        "Keypoint AP@0.75": keypoint_map_75,
        "Keypoint AR": keypoint_mar,
        "F1 / Precision / Recall": f1_precision_recall,
    }
    return {name: columns for name, columns in metric_groups.items() if columns}


def _read_metrics_csv(metrics_csv: str) -> tuple[Any, Any]:
    """Read a PTL CSVLogger metrics file and return both step-level and epoch-averaged DataFrames.

    Returns:
        A ``(raw_df, epoch_df)`` pair where ``raw_df`` contains every logged row
        (one per training step or validation epoch) and ``epoch_df`` is the per-epoch
        mean used for column detection and log-scale checks.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for training metric plots. Install it with: pip install pandas") from exc

    csv_path = Path(metrics_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        raise ValueError("metrics.csv does not contain an 'epoch' column.")
    raw_df = _drop_trailing_validation_only_epochs(df)
    epoch_df = raw_df.groupby("epoch").mean(numeric_only=True).reset_index()
    return raw_df, epoch_df


def _drop_trailing_validation_only_epochs(df: Any) -> Any:
    """Remove post-fit validation rows that Lightning logs as a synthetic final epoch.

    The Roboflow finetune demos run ``trainer.validate(...)`` after ``trainer.fit(...)`` to write a final metrics JSON.
    PTL appends that validation pass to the same ``CSVLogger`` file using ``epoch == max_epochs``. That row is useful as
    a standalone final validation result, but it is not part of the training curve and can create a misleading last-
    epoch jump in plots. Only trailing epochs with validation/test metrics and no training metrics are removed, and only
    when the CSV also contains real training rows. Pure validation CSV files are preserved.
    """
    train_columns = [column for column in df.columns if str(column).startswith("train/")]
    eval_columns = [column for column in df.columns if str(column).startswith(("val/", "test/"))]
    if not train_columns or not eval_columns:
        return df
    if not df[train_columns].notna().any(axis=None):
        return df

    cleaned = df
    while len(cleaned) > 0:
        last_epoch = cleaned["epoch"].iloc[-1]
        epoch_mask = cleaned["epoch"] == last_epoch
        epoch_rows = cleaned.loc[epoch_mask]
        if cleaned.loc[~epoch_mask].empty:
            return cleaned
        has_train_metrics = epoch_rows[train_columns].notna().any(axis=None)
        has_eval_metrics = epoch_rows[eval_columns].notna().any(axis=None)
        if has_train_metrics or not has_eval_metrics:
            return cleaned
        cleaned = cleaned.loc[~epoch_mask]
    return cleaned


def _plot_metric_groups(
    raw_df: Any,
    epoch_df: Any,
    metric_groups: dict[str, list[str]],
    *,
    title: str,
    output_path: str | None,
    loss_log_scale: bool,
) -> Figure:
    """Build a figure for grouped metrics."""
    try:
        import matplotlib

        _ensure_headless_backend(matplotlib)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for training metric plots. Install it with: pip install matplotlib"
        ) from exc

    if not metric_groups:
        raise ValueError("metrics.csv does not contain any supported non-empty metric columns.")

    n_groups = len(metric_groups)
    n_cols = 1 if n_groups == 1 else 2
    n_rows = (n_groups + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, (subplot_title, metric_list) in enumerate(metric_groups.items()):
        ax = axes_flat[idx]
        _plot_columns_on_axes(ax, raw_df, epoch_df, metric_list)
        ax.set_title(subplot_title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(subplot_title, fontsize=11)
        ax.grid(True, alpha=0.3)
        if subplot_title == "Loss" and loss_log_scale:
            group_data = epoch_df[metric_list]
            if (group_data <= 0).any(axis=None):
                warnings.warn(
                    "loss_log_scale=True was requested, but at least one loss value is non-positive; "
                    "using linear scale for the Loss panel.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                ax.set_yscale("log")
        if subplot_title == "Loss":
            _place_legend_below_axes(ax)

    for idx in range(n_groups, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if "Loss" in metric_groups:
        fig.subplots_adjust(bottom=0.24 if n_groups == 1 else 0.12)
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_loss_metrics(
    metrics_csv: str,
    output_path: str | None = None,
    loss_log_scale: bool = False,
) -> Figure:
    """Plot aggregate and component training losses from a PTL ``metrics.csv`` file.

    Reads the CSV written by PyTorch Lightning's ``CSVLogger``, groups loss columns
    (aggregate loss, per-component scalars, and keypoint NLL terms) into a single panel,
    and renders an optional seaborn error-band overlay when seaborn is installed.

    Args:
        metrics_csv: Path to the ``metrics.csv`` written by PTL's ``CSVLogger``.
        output_path: Optional filesystem path to save the rendered figure (PNG, PDF, …).
            When ``None`` the figure is returned but not written to disk.
        loss_log_scale: When ``True``, the loss y-axis uses a logarithmic scale.
            Useful when loss components span several orders of magnitude.

    Returns:
        A ``matplotlib.figure.Figure`` containing the loss panel.

    Raises:
        FileNotFoundError: When ``metrics_csv`` does not exist.
        ImportError: When ``pandas`` or ``matplotlib`` is not installed.
        ValueError: When ``metrics_csv`` contains no loss columns.

    Examples:
        .. code-block:: python

            from mmdet.models.layers.rf_detr.visualize.training import plot_loss_metrics
            fig = plot_loss_metrics("output/rfdetr_small/metrics.csv")
            fig = plot_loss_metrics("output/rfdetr_small/metrics.csv", output_path="loss.png", loss_log_scale=True)
    """
    raw_df, epoch_df = _read_metrics_csv(metrics_csv)
    groups = _build_metric_groups(epoch_df)
    return _plot_metric_groups(
        raw_df,
        epoch_df,
        {"Loss": groups["Loss"]} if "Loss" in groups else {},
        title="RF-DETR Loss Metrics",
        output_path=output_path,
        loss_log_scale=loss_log_scale,
    )


def plot_map_metrics(
    metrics_csv: str,
    output_path: str | None = None,
) -> Figure:
    """Plot train/val/test detection and keypoint mAP metrics from a PTL ``metrics.csv`` file.

    Reads the CSV written by PyTorch Lightning's ``CSVLogger``, selects all AP-family
    columns (AP@0.50, AP@0.75, AP@0.50:0.95 for both detection and keypoints), and renders
    them in a single combined panel.  EMA series are included when present so both live and
    EMA metric trajectories are visible in the legend.

    Args:
        metrics_csv: Path to the ``metrics.csv`` written by PTL's ``CSVLogger``.
        output_path: Optional filesystem path to save the rendered figure (PNG, PDF, …).
            When ``None`` the figure is returned but not written to disk.

    Returns:
        A ``matplotlib.figure.Figure`` containing the combined mAP panel.

    Raises:
        FileNotFoundError: When ``metrics_csv`` does not exist.
        ImportError: When ``pandas`` or ``matplotlib`` is not installed.
        ValueError: When ``metrics_csv`` contains no supported mAP metric columns.

    Examples:
        .. code-block:: python

            from mmdet.models.layers.rf_detr.visualize.training import plot_map_metrics
            fig = plot_map_metrics("output/rfdetr_small/metrics.csv")
            fig = plot_map_metrics("output/rfdetr_small/metrics.csv", output_path="map.png")
    """
    raw_df, epoch_df = _read_metrics_csv(metrics_csv)
    metric_groups = _build_metric_groups(epoch_df)
    map_columns = [
        column
        for group_name, columns in metric_groups.items()
        if group_name
        in {
            "Detection AP@0.50",
            "Detection AP@0.50:0.95",
            "Detection AP@0.75",
            "Keypoint AP@0.50",
            "Keypoint AP@0.50:0.95",
            "Keypoint AP@0.75",
        }
        for column in columns
    ]
    if not map_columns:
        raise ValueError("metrics.csv does not contain any supported non-empty mAP metric columns.")
    return _plot_map_columns(raw_df, epoch_df, map_columns, output_path=output_path)


def _plot_map_columns(raw_df: Any, epoch_df: Any, metric_columns: list[str], *, output_path: str | None) -> Figure:
    """Plot mAP metrics on a single axes with line style by split."""
    try:
        import matplotlib

        _ensure_headless_backend(matplotlib)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for mAP metric plots. Install it with: pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=(12, 6))
    _plot_columns_on_axes(ax, raw_df, epoch_df, metric_columns)

    ax.set_title("RF-DETR mAP Metrics", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("mAP", fontsize=11)
    ax.grid(True, alpha=0.3)
    _place_legend_below_axes(ax)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metrics(
    metrics_csv: str,
    output_path: str | None = None,
    loss_log_scale: bool = False,
) -> Figure:
    """Read a PTL ``CSVLogger`` metrics file and build a training plot.

    The figure contains one subplot per metric group (loss, detection metrics,
    keypoint metrics, and F1/precision/recall), arranged in a 2-column grid.
    Only groups with at least one non-NaN column are shown.

    When seaborn is installed, each series is drawn as mean ± 1 std-dev band
    computed from the within-epoch step-level rows logged by PTL's
    ``CSVLogger``.  Metrics recorded only once per epoch (e.g. ``val/mAP_50``)
    show a plain line because their per-epoch std is zero.  When seaborn is
    absent the plot falls back to epoch-averaged lines.

    Args:
        metrics_csv: Path to the ``metrics.csv`` file produced by
            ``CSVLogger``.
        output_path: Optional destination for the PNG file. If omitted, the
            figure is returned without saving.
        loss_log_scale: If ``True``, use a logarithmic y-axis for the Loss
            panel when all loss values are positive.

    Returns:
        The matplotlib figure. The figure is left open so notebook cells can
        display it inline.

    Raises:
        ImportError: If ``matplotlib`` or ``pandas`` are not installed.
        FileNotFoundError: If ``metrics_csv`` does not exist.
    """
    raw_df, epoch_df = _read_metrics_csv(metrics_csv)
    metric_groups = _build_metric_groups(epoch_df)
    return _plot_metric_groups(
        raw_df,
        epoch_df,
        metric_groups,
        title="RF-DETR Training Metrics",
        output_path=output_path,
        loss_log_scale=loss_log_scale,
    )
