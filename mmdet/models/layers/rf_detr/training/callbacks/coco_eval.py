# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""COCOEvalCallback — torchmetrics-based mAP and F1 evaluation."""

import contextlib
import io
import logging
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
from pytorch_lightning import Callback
from torchmetrics.detection import MeanAveragePrecision

from mmdet.models.layers.rf_detr.datasets import get_coco_api_from_dataset
from mmdet.models.layers.rf_detr.evaluation.f1_sweep import sweep_confidence_thresholds
from mmdet.models.layers.rf_detr.evaluation.keypoint_oks import (
    DEFAULT_KEYPOINT_MAX_DETS,
    MetricKeypointOKS,
    OKSKey,
)
from mmdet.models.layers.rf_detr.evaluation.matching import (
    build_matching_data,
    distributed_merge_matching_data,
    init_matching_accumulator,
    merge_matching_data,
)
from mmdet.models.layers.rf_detr.utilities.box_ops import box_cxcywh_to_xyxy
from mmdet.models.layers.rf_detr.utilities.console import (
    _IS_RICH_AVAILABLE,
    _get_rich_console,
    _has_progress_bar,
    _render_overall_merged,
    _render_summary_tables,
)
from mmdet.models.layers.rf_detr.utilities.distributed import all_gather, get_world_size, is_dist_avail_and_initialized
from mmdet.models.layers.rf_detr.utilities.logger import get_logger

logger = get_logger()


def _warn_missing_rich_once(warning_emitted: bool) -> bool:
    """Warn once when metric table rendering is skipped because Rich is unavailable.

    Args:
        warning_emitted: Whether this warning has already been emitted.

    Returns:
        Always ``True``; caller assigns back to suppress future warnings.
    """
    if warning_emitted:
        return True
    logger.warning("Rich is not installed; skipping metric table rendering. Install `rich` to enable tables.")
    return True


def _get_ema_inner_module(ema_cb: Any) -> Any:
    """Return the inner ``nn.Module`` wrapped by an EMA callback.

    ``RFDETREMACallback._average_model`` is a private attribute holding a ``torch.optim.swa_utils.AveragedModel``
    (which exposes the actual module on ``.module``).  This helper centralises the access so that consumers degrade
    gracefully when the EMA model has not yet been initialised — preferable to reaching through two layers of
    private attributes at every call site.

    Args:
        ema_cb: EMA callback instance (or ``None``).

    Returns:
        The inner module wrapped by ``AveragedModel``, or ``None`` when no EMA model is available.
    """
    if ema_cb is None:
        return None
    averaged = getattr(ema_cb, "_average_model", None)
    if averaged is None:
        return None
    return getattr(averaged, "module", averaged)


class COCOEvalCallback(Callback):
    """Validation callback that computes mAP (via torchmetrics) and macro-F1.

    Accumulates predictions and targets across validation batches, then at epoch end computes:

    - ``val/mAP_50_95``, ``val/mAP_50``, ``val/mAP_75``, ``val/mAR`` using
      ``torchmetrics.detection.MeanAveragePrecision``.
    - Per-class ``val/AP/<name>`` when class names are available.
    - ``val/F1``, ``val/precision``, ``val/recall`` from a confidence-threshold
      sweep over compact per-class matching data (DDP-safe).

    For segmentation models (``segmentation=True``) additional metrics ``val/segm_mAP_50_95`` and ``val/segm_mAP_50``
    are logged.

    Args:
        max_dets: Maximum detections per image passed to
            ``MeanAveragePrecision``. Defaults to :data:`~rfdetr.evaluation.keypoint_oks.DEFAULT_KEYPOINT_MAX_DETS`.
        segmentation: When ``True``, evaluate both bbox and segm IoU using
            ``backend="faster_coco_eval"``. Defaults to ``False``.
        eval_interval: Run validation metrics every N epochs. Test metrics are
            always computed when ``trainer.test()`` is called.
        log_per_class_metrics: When ``False``, skip per-class AP logging/table.
    """

    def __init__(
        self,
        max_dets: int = DEFAULT_KEYPOINT_MAX_DETS,
        segmentation: bool = False,
        eval_interval: int = 1,
        log_per_class_metrics: bool = True,
        keypoint_oks_sigmas: list[float] | None = None,
        in_notebook: bool | None = None,
    ) -> None:
        super().__init__()
        self._max_dets = max_dets
        self._segmentation = segmentation
        self._eval_interval = max(1, int(eval_interval))
        self._log_per_class_metrics = bool(log_per_class_metrics)
        self._class_names: list[str] = []
        self._cat_id_to_name: dict[int, str] = {}
        self._f1_local: dict[int, dict[str, Any]] = init_matching_accumulator()
        self._f1_train_local: dict[int, dict[str, Any]] = init_matching_accumulator()
        # Whether the EMA metric received ≥1 update this epoch.  Gates the EMA cross-rank
        # sync so it is issued symmetrically on all DDP ranks (see _should_compute_ema).
        self._ema_has_updates: bool = False
        self._missing_rich_warning_emitted: bool = False
        self._output_widget: Any = None  # ipywidgets.Output, created lazily
        self._keypoint_mode: bool = False
        self._use_segm_metrics: bool = segmentation
        self._train_segm_skip_warned: bool = False
        self._keypoint_oks_metrics: dict[str, MetricKeypointOKS] = {}
        self._keypoint_oks_sigmas = keypoint_oks_sigmas
        self._in_notebook: bool = False
        if in_notebook is None:
            with contextlib.suppress(ImportError):
                from IPython import get_ipython

                self._in_notebook = get_ipython() is not None
        else:
            self._in_notebook = in_notebook

    # ------------------------------------------------------------------
    # PTL lifecycle hooks
    # ------------------------------------------------------------------

    def setup(self, trainer: Any, pl_module: Any, stage: str) -> None:
        """Instantiate ``MeanAveragePrecision`` after DDP device placement.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            stage: One of ``"fit"``, ``"validate"``, ``"test"``, ``"predict"``.
        """
        model_config = getattr(pl_module, "model_config", None)
        # Some callback unit shims omit model_config; missing keypoint flag means bbox/segm evaluation.
        use_grouppose_keypoints = (
            getattr(model_config, "use_grouppose_keypoints", False) if model_config is not None else False
        )
        self._keypoint_mode = use_grouppose_keypoints is True
        self._use_segm_metrics = self._segmentation and not self._keypoint_mode
        iou_type: Any = ["bbox", "segm"] if self._use_segm_metrics else "bbox"
        kwargs: dict[str, Any] = dict(
            class_metrics=True,
            max_detection_thresholds=[1, 10, self._max_dets],
            # Disable torchmetrics' built-in cross-rank sync: its `gather_all_tensors` requires every
            # state tensor to have the same ndim on all ranks, but DDP seg validation produces
            # per-rank states that are scalar on some ranks and vectors on others, so the internal
            # sync issues a different number of collectives per rank and deadlocks (known torchmetrics
            # bug, #931/#449). We merge state across ranks ourselves in `_merge_metric_state_across_ranks`
            # using the repo's fixed-shape `all_gather`, then compute() runs locally on the full set.
            sync_on_compute=False,
        )
        kwargs["backend"] = "faster_coco_eval"
        self.map_metric = MeanAveragePrecision(iou_type=iou_type, **kwargs)
        self.map_metric_train = MeanAveragePrecision(iou_type=iou_type, **kwargs)
        # Verify _MAP_STATE_ATTRS is complete for the installed torchmetrics version.  A missing
        # attr is silently skipped in _merge_metric_state_across_ranks, producing wrong mAP with
        # no error — an upgrade that adds a list-type state would hit this silently without the check.
        installed = {k for k, v in self.map_metric._defaults.items() if isinstance(v, list)}
        declared = set(self._MAP_STATE_ATTRS)
        if installed != declared:
            raise RuntimeError(
                "COCOEvalCallback._MAP_STATE_ATTRS is out of sync with the installed torchmetrics"
                f" (version {self.map_metric.__class__.__module__})."
                f" Missing from _MAP_STATE_ATTRS: {sorted(installed - declared)}."
                f" Stale in _MAP_STATE_ATTRS: {sorted(declared - installed)}."
                ' Re-run: python -c "from torchmetrics.detection import MeanAveragePrecision;'
                " m = MeanAveragePrecision();"
                ' print(sorted(k for k, v in m._defaults.items() if isinstance(v, list)))"'
                " and update COCOEvalCallback._MAP_STATE_ATTRS to match."
            )
        # Separate metric for the EMA model.  Created deterministically on EVERY rank in
        # on_validation_epoch_start / on_test_epoch_start (see _prepare_ema_metric) so its
        # cross-rank compute() sync is issued symmetrically and cannot deadlock DDP val.
        self.map_metric_ema: Any = None

    def teardown(self, trainer: Any, pl_module: Any, stage: str) -> None:
        """Release the notebook output widget when the trainer exits.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            stage: One of ``"fit"``, ``"validate"``, ``"test"``, ``"predict"``.
        """
        self._output_widget = None

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Pull class names from the DataModule once the datasets are set up.

        Builds a ``category_id → name`` mapping from the COCO annotation metadata so that per-class AP is logged under
        the class name regardless of whether the dataset uses sequential or non-sequential category IDs.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        dm = trainer.datamodule
        if dm is None:
            return
        if hasattr(dm, "class_names"):
            self._class_names = dm.class_names or []
        # Build cat_id → name from the COCO annotation object when available.
        for attr in ("_dataset_train", "_dataset_val"):
            dataset = getattr(dm, attr, None)
            if dataset is None:
                continue
            coco = getattr(dataset, "coco", None)
            if coco is not None and hasattr(coco, "cats"):
                if hasattr(coco, "label2cat"):
                    # remap_category_ids=True: dataset labels are 0-based contiguous
                    # indices.  label2cat maps remapped_label → original_cat_id;
                    # use it to build label → name so class IDs match predictions.
                    self._cat_id_to_name = {
                        label: coco.cats[cat_id]["name"] for label, cat_id in coco.label2cat.items()
                    }
                else:
                    # Raw COCO category IDs used as labels (standard COCO dataset).
                    self._cat_id_to_name = {k: v["name"] for k, v in coco.cats.items()}
                return
        # Fallback: treat class_names as 0-based sequential labels.
        self._cat_id_to_name = {i: name for i, name in enumerate(self._class_names)}

    def on_validation_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        """Prepare the EMA metric on every rank before validation (keeps DDP collectives symmetric).

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        self.map_metric.reset()
        self._f1_local = init_matching_accumulator()
        self._reset_keypoint_split("val")
        self._reset_keypoint_split("val_ema")
        self._prepare_ema_metric(trainer, pl_module)

    def on_test_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        """Reset ``_ema_has_updates`` before test to prevent stale validation state from triggering EMA compute.

        ``on_test_batch_end`` never sets ``_ema_has_updates = True``, so EMA compute is always skipped during
        test (test metrics already reflect the EMA model via checkpoint loading in
        :class:`~rfdetr.training.callbacks.best_model.BestModelCallback`).  Without this hook a stale ``True`` value
        left by a preceding validation epoch would make ``_should_compute_ema`` return ``True``, causing an
        empty-state EMA compute pass that logs sentinel ``-1`` values.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        self.map_metric.reset()
        self._f1_local = init_matching_accumulator()
        self._reset_keypoint_split("test")
        self._prepare_ema_metric(trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Accumulate train predictions for optional train-split mAP logging.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            outputs: Return value of ``training_step``.
            batch: The device-transferred batch (unused here).
            batch_idx: Batch index within the training epoch.
        """
        if getattr(getattr(pl_module, "train_config", None), "compute_train_metrics", False) is not True:
            return
        if not isinstance(outputs, dict) or "results" not in outputs or "targets" not in outputs:
            return

        preds: list[dict[str, torch.Tensor]] = self._convert_preds(outputs["results"])
        targets = self._convert_targets(outputs["targets"])
        # In training mode pred_masks is a sparse dict, excluded from postprocess inputs, so
        # preds have no masks key.  torchmetrics requires it when iou_type includes "segm" → skip.
        if self._use_segm_metrics and preds and "masks" not in preds[0]:
            if not self._train_segm_skip_warned:
                logger.info(
                    "Train-split segmentation mAP skipped: pred_masks is a sparse dict during training "
                    "(sparse_forward).  Only val/test segm mAP is available."
                )
                self._train_segm_skip_warned = True
            return
        self.map_metric_train.update(preds, targets)

        iou_type = "segm" if self._use_segm_metrics else "bbox"
        batch_matching = build_matching_data(preds, targets, iou_threshold=0.5, iou_type=iou_type)
        merge_matching_data(self._f1_train_local, batch_matching)
        self._update_keypoint_oks_metric(trainer, outputs, split="train")

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Compute optional train-split mAP at the end of the training epoch.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        if getattr(getattr(pl_module, "train_config", None), "compute_train_metrics", False) is not True:
            self.map_metric_train.reset()
            self._f1_train_local = init_matching_accumulator()
            self._reset_keypoint_split("train")
            return
        if self._eval_interval > 1:
            current_epoch = int(getattr(trainer, "current_epoch", 0)) + 1
            max_epochs = getattr(trainer, "max_epochs", None)
            is_last_epoch = isinstance(max_epochs, int) and max_epochs > 0 and current_epoch >= max_epochs
            if current_epoch % self._eval_interval != 0 and not is_last_epoch:
                self.map_metric_train.reset()
                self._f1_train_local = init_matching_accumulator()
                self._reset_keypoint_split("train")
                return
        self._compute_and_log(trainer, pl_module, "train", metric=self.map_metric_train)

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Accumulate predictions and matching data for one validation batch.

        Expects ``outputs`` to be the dict returned by ``RFDETRModelModule.validation_step``: ``{"results": list[dict],
        "targets": list[dict]}``.

        When an EMA callback is present the EMA model is run on the same batch in a separate ``torch.no_grad()`` forward
        pass so that base and EMA metrics are computed from independent predictions.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            outputs: Return value of ``validation_step``.
            batch: The device-transferred batch ``(samples, targets)``.
            batch_idx: Batch index within the validation epoch.
        """
        preds: list[dict[str, torch.Tensor]] = self._convert_preds(outputs["results"])
        targets = self._convert_targets(outputs["targets"])

        self.map_metric.update(preds, targets)

        iou_type = "segm" if self._use_segm_metrics else "bbox"
        batch_matching = build_matching_data(preds, targets, iou_threshold=0.5, iou_type=iou_type)
        merge_matching_data(self._f1_local, batch_matching)
        self._update_keypoint_oks_metric(trainer, outputs, split="val")

        # Run EMA model separately on the same batch so that base and EMA metrics
        # are computed from independent forward passes rather than being aliases.
        # The EMA metric object itself is created on every rank in
        # on_validation_epoch_start (_prepare_ema_metric); here we only run the EMA
        # forward pass + update when the averaged model is available.  ema_cb._average_model
        # availability is rank-invariant (EMA updates fire on the same global step on every
        # rank), so per-rank EMA update counts stay consistent.
        ema_cb = self._get_ema_callback(trainer)
        ema_inner = _get_ema_inner_module(ema_cb)
        if ema_cb is not None and ema_inner is not None and self.map_metric_ema is not None:
            samples, _ = batch
            orig_sizes = torch.stack([t["orig_size"] for t in outputs["targets"]]).to(pl_module.device)
            ema_underlying = ema_inner.model
            with torch.no_grad():
                ema_underlying.eval()  # AveragedModel deepcopy is not managed by PTL
                ema_outputs = ema_underlying(samples)
                ema_results = pl_module.postprocess(ema_outputs, orig_sizes)
            ema_preds = self._convert_preds(ema_results)
            self.map_metric_ema.update(ema_preds, targets)
            self._update_keypoint_oks_metric(
                trainer,
                {"results": ema_results, "targets": outputs["targets"]},
                split="val_ema",
            )
            self._ema_has_updates = True

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Compute and log mAP and F1 metrics at the end of the validation epoch.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        if self._eval_interval > 1:
            current_epoch = int(getattr(trainer, "current_epoch", 0)) + 1
            max_epochs = getattr(trainer, "max_epochs", None)
            is_last_epoch = isinstance(max_epochs, int) and max_epochs > 0 and current_epoch >= max_epochs
            if current_epoch % self._eval_interval != 0 and not is_last_epoch:
                self.map_metric.reset()
                if self.map_metric_ema is not None:
                    self.map_metric_ema.reset()
                self._f1_local = init_matching_accumulator()
                self._reset_keypoint_split("val")
                self._reset_keypoint_split("val_ema")
                return
        self._compute_and_log(trainer, pl_module, "val")

    def on_test_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate predictions and matching data for one test batch.

        Mirrors :meth:`on_validation_batch_end` for the test evaluation loop triggered by ``trainer.test()`` at the end
        of training.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            outputs: Return value of ``test_step``.
            batch: Raw batch (unused here).
            batch_idx: Batch index within the test epoch.
            dataloader_idx: Index of the test dataloader (unused here).
        """
        preds: list[dict[str, torch.Tensor]] = self._convert_preds(outputs["results"])
        targets = self._convert_targets(outputs["targets"])

        self.map_metric.update(preds, targets)

        iou_type = "segm" if self._use_segm_metrics else "bbox"
        batch_matching = build_matching_data(preds, targets, iou_threshold=0.5, iou_type=iou_type)
        merge_matching_data(self._f1_local, batch_matching)
        self._update_keypoint_oks_metric(trainer, outputs, split="test")

    def on_test_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Compute and log mAP and F1 under ``test/`` prefix at end of test epoch.

        Mirrors :meth:`on_validation_epoch_end` for the test evaluation loop.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        self._compute_and_log(trainer, pl_module, "test")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_and_log(self, trainer: Any, pl_module: Any, split: str, *, metric: Any | None = None) -> None:
        """Shared epoch-end logic for validation and test evaluation loops.

        Computes mAP (via ``self.map_metric``), runs the F1 confidence-threshold sweep, logs all scalar metrics via
        ``pl_module.log``, prints two summary tables to the terminal, and resets internal accumulators.  When
        ``self.map_metric_ema`` is set, EMA variants of all metrics (including ``ema_segm_mAP_50_95`` and
        ``ema_segm_mAP_50`` for segmentation models) are logged under the same ``split/`` namespace.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            split: Metric namespace — ``"val"`` or ``"test"``.
            metric: Optional split-specific mAP accumulator. Defaults to the validation/test accumulator.
        """
        metric = self.map_metric if metric is None else metric
        f1_local = self._f1_train_local if split == "train" else self._f1_local
        if not self._metric_has_updates(metric):
            metric.reset()
            self._reset_f1_local(split)
            self._reset_keypoint_split(split)
            logger.debug("Skipping %s COCO metric compute because no predictions were accumulated.", split)
            return

        # Merge per-rank state across ranks ourselves (DDP-safe, fixed-shape gather) before the
        # metric computes locally — replaces torchmetrics' deadlock-prone internal sync. No-op when
        # not distributed. Called unconditionally on every rank, so the collectives stay symmetric.
        self._merge_metric_state_across_ranks(metric)
        metrics = self._compute_map_metric(trainer, metric)

        # torchmetrics prefixes all keys when iou_type is a list (e.g. "bbox_map")
        pfx = "bbox_" if self._use_segm_metrics else ""
        mar_key = f"{pfx}mar_{self._max_dets}"

        overall: dict[str, float] = {
            "mAP 50:95": float(metrics[f"{pfx}map"]),
            "mAP 50": float(metrics[f"{pfx}map_50"]),
            "mAP 75": float(metrics[f"{pfx}map_75"]),
            f"mAR @{self._max_dets}": float(metrics[mar_key]),
        }

        pl_module.log(
            f"{split}/mAP_50_95", metrics[f"{pfx}map"], prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        pl_module.log(
            f"{split}/mAP_50", metrics[f"{pfx}map_50"], prog_bar=True, logger=True, on_step=False, on_epoch=True
        )
        pl_module.log(f"{split}/mAP_75", metrics[f"{pfx}map_75"], logger=True, on_step=False, on_epoch=True)
        pl_module.log(f"{split}/mAR", metrics[mar_key], logger=True, on_step=False, on_epoch=True)

        # Write directly into callback_metrics so ModelCheckpoint / EarlyStopping
        # read fresh values each epoch.  pl_module.log() from a callback's
        # on_*_epoch_end goes only to logged_metrics (external loggers), not to
        # callback_metrics, so checkpointing would see stale values otherwise.
        trainer.callback_metrics[f"{split}/mAP_50_95"] = metrics[f"{pfx}map"].detach().cpu()
        trainer.callback_metrics[f"{split}/mAP_50"] = metrics[f"{pfx}map_50"].detach().cpu()
        trainer.callback_metrics[f"{split}/mAP_75"] = metrics[f"{pfx}map_75"].detach().cpu()
        trainer.callback_metrics[f"{split}/mAR"] = metrics[mar_key].detach().cpu()

        # EMA metrics — computed from a separate EMA forward pass accumulated in
        # on_validation_batch_end, so base and EMA values are independent.  The EMA
        # compute() triggers a cross-rank metric sync, so it must be issued by EVERY rank
        # or none: a rank whose EMA metric is empty/absent would otherwise skip this
        # collective and desync the DDP collective sequence, deadlocking validation
        # (#931 / #449).  _should_compute_ema makes the decision unanimous across ranks.
        should_compute_ema = self._should_compute_ema(pl_module)
        if should_compute_ema:
            self._merge_metric_state_across_ranks(self.map_metric_ema)
            ema_metrics = self._compute_map_metric(trainer, self.map_metric_ema)
            pl_module.log(
                f"{split}/ema_mAP_50_95",
                ema_metrics[f"{pfx}map"],
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
            )
            pl_module.log(f"{split}/ema_mAP_50", ema_metrics[f"{pfx}map_50"], logger=True, on_step=False, on_epoch=True)
            pl_module.log(f"{split}/ema_mAR", ema_metrics[mar_key], logger=True, on_step=False, on_epoch=True)
            trainer.callback_metrics[f"{split}/ema_mAP_50_95"] = ema_metrics[f"{pfx}map"].detach().cpu()
            trainer.callback_metrics[f"{split}/ema_mAP_50"] = ema_metrics[f"{pfx}map_50"].detach().cpu()
            trainer.callback_metrics[f"{split}/ema_mAR"] = ema_metrics[mar_key].detach().cpu()
            if self._use_segm_metrics:
                pl_module.log(
                    f"{split}/ema_segm_mAP_50_95", ema_metrics["segm_map"], logger=True, on_step=False, on_epoch=True
                )
                pl_module.log(
                    f"{split}/ema_segm_mAP_50", ema_metrics["segm_map_50"], logger=True, on_step=False, on_epoch=True
                )
                trainer.callback_metrics[f"{split}/ema_segm_mAP_50_95"] = ema_metrics["segm_map"].detach().cpu()
                trainer.callback_metrics[f"{split}/ema_segm_mAP_50"] = ema_metrics["segm_map_50"].detach().cpu()
            self.map_metric_ema.reset()
        elif self.map_metric_ema is not None:
            # Not all ranks have EMA data this epoch (e.g. EMA not yet warmed up) → skip the
            # sync uniformly on every rank, but clear local state so the next epoch is clean.
            self.map_metric_ema.reset()

        if self._use_segm_metrics:
            overall["segm mAP 50:95"] = float(metrics["segm_map"])
            overall["segm mAP 50"] = float(metrics["segm_map_50"])
            pl_module.log(f"{split}/segm_mAP_50_95", metrics["segm_map"], logger=True, on_step=False, on_epoch=True)
            pl_module.log(f"{split}/segm_mAP_50", metrics["segm_map_50"], logger=True, on_step=False, on_epoch=True)
            trainer.callback_metrics[f"{split}/segm_mAP_50_95"] = metrics["segm_map"].detach().cpu()
            trainer.callback_metrics[f"{split}/segm_mAP_50"] = metrics["segm_map_50"].detach().cpu()

        # F1 sweep — run first so per-class F1/prec/rec are available when
        # building the unified per-class table rows below.
        merged = distributed_merge_matching_data(f1_local)
        # category_id → {f1, precision, recall} at the best macro-F1 threshold
        f1_by_cid: dict[int, dict[str, float]] = {}
        if merged:
            sorted_ids = sorted(merged.keys())
            per_class_list = [merged[cid] for cid in sorted_ids]
            classes_with_gt = [i for i, cid in enumerate(sorted_ids) if merged[cid]["total_gt"] > 0]
            f1_results = sweep_confidence_thresholds(per_class_list, np.linspace(0, 1, 101), classes_with_gt)
            best = max(f1_results, key=lambda x: x["macro_f1"])
            overall["F1"] = float(best["macro_f1"])
            overall["Precision"] = float(best["macro_precision"])
            overall["Recall"] = float(best["macro_recall"])
            pl_module.log(
                f"{split}/F1",
                float(best["macro_f1"]),
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
            )
            pl_module.log(
                f"{split}/precision", float(best["macro_precision"]), logger=True, on_step=False, on_epoch=True
            )
            pl_module.log(f"{split}/recall", float(best["macro_recall"]), logger=True, on_step=False, on_epoch=True)
            trainer.callback_metrics[f"{split}/F1"] = torch.tensor(float(best["macro_f1"]))
            trainer.callback_metrics[f"{split}/precision"] = torch.tensor(float(best["macro_precision"]))
            trainer.callback_metrics[f"{split}/recall"] = torch.tensor(float(best["macro_recall"]))
            for k, cid in enumerate(sorted_ids):
                f1_by_cid[cid] = {
                    "f1": float(best["per_class_f1"][k]),
                    "precision": float(best["per_class_prec"][k]),
                    "recall": float(best["per_class_rec"][k]),
                }
        else:
            overall["F1"] = 0.0
            overall["Precision"] = 0.0
            overall["Recall"] = 0.0
            pl_module.log(f"{split}/F1", 0.0, prog_bar=True, logger=True, on_step=False, on_epoch=True)
            pl_module.log(f"{split}/precision", 0.0, logger=True, on_step=False, on_epoch=True)
            pl_module.log(f"{split}/recall", 0.0, logger=True, on_step=False, on_epoch=True)
            trainer.callback_metrics[f"{split}/F1"] = torch.tensor(0.0)
            trainer.callback_metrics[f"{split}/precision"] = torch.tensor(0.0)
            trainer.callback_metrics[f"{split}/recall"] = torch.tensor(0.0)

        # torchmetrics returns `classes` as a 0-d scalar when only one class is
        # present in the batch.  Ensure it is always 1-d before iterating.
        if "classes" in metrics and metrics["classes"].ndim == 0:
            metrics = dict(metrics)
            metrics["classes"] = metrics["classes"].unsqueeze(0)
            for k in list(metrics):
                if isinstance(metrics[k], torch.Tensor) and metrics[k].ndim == 0 and "per_class" in k:
                    metrics[k] = metrics[k].unsqueeze(0)

        # Per-class AR from torchmetrics (keyed by category_id)
        ar_pc_key = f"{pfx}mar_{self._max_dets}_per_class"
        ar_by_cid: dict[int, float] = {}
        if ar_pc_key in metrics and "classes" in metrics:
            for class_id, ar in zip(metrics["classes"], metrics[ar_pc_key]):
                ar_by_cid[int(class_id)] = float(ar)

        # Unified per-class rows: AP 50:95 | AR | F1 | Precision | Recall
        # Classes with no ground-truth annotations are skipped (pycocotools
        # returns -1 for AP and torchmetrics returns NaN for AR on such classes,
        # so they would show as all dashes in the table).
        per_class = self._build_per_class_rows(
            metrics=metrics, pfx=pfx, split=split, pl_module=pl_module, ar_by_cid=ar_by_cid, f1_by_cid=f1_by_cid
        )

        self._print_metrics_tables(trainer, split, overall, per_class)
        self._compute_and_log_keypoint_map(split, pl_module, trainer)
        if split == "val" and should_compute_ema:
            self._compute_and_log_keypoint_map("val_ema", pl_module, trainer, log_split="val", metric_prefix="ema_")
        elif split == "val":
            self._reset_keypoint_split("val_ema")
        metric.reset()
        self._reset_f1_local(split)

    def _reset_f1_local(self, split: str) -> None:
        """Reset the F1 accumulator for a metric split."""
        if split == "train":
            self._f1_train_local = init_matching_accumulator()
        else:
            self._f1_local = init_matching_accumulator()

    def _get_ema_callback(self, trainer: Any) -> Any:
        """Return the EMA callback instance, or ``None`` if not present."""
        for callback in getattr(trainer, "callbacks", []):
            if callable(getattr(callback, "get_ema_model_state_dict", None)):
                return callback
        return None

    def _compute_map_metric(self, trainer: Any, metric: Any) -> dict[str, Any]:
        """Compute a torchmetrics mAP metric while suppressing duplicate terminal summaries under progress bars."""
        if not _has_progress_bar(trainer):
            return metric.compute()

        metric_loggers = (logger, logging.getLogger("faster_coco_eval"), logging.getLogger("faster_coco_eval.core"))
        previous_levels = [(metric_logger, metric_logger.level) for metric_logger in metric_loggers]
        try:
            for metric_logger in metric_loggers:
                if metric_logger.getEffectiveLevel() < logging.WARNING:
                    metric_logger.setLevel(logging.WARNING)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return metric.compute()
        finally:
            for metric_logger, previous_level in previous_levels:
                metric_logger.setLevel(previous_level)

    def _prepare_ema_metric(self, trainer: Any, pl_module: Any) -> None:
        """Ensure ``map_metric_ema`` exists (and is reset) on EVERY rank when EMA is active.

        Driven by the rank-invariant presence of the EMA callback rather than by per-batch state, so any cross-rank
        state merge (via :meth:`_merge_metric_state_across_ranks`) is issued symmetrically across DDP ranks. Previously
        the metric was created lazily in :meth:`on_validation_batch_end`, so a rank with an empty/uneven shard could
        finish without it, skip the merge/compute path, and deadlock validation (#931 / #449).

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule (provides the device for metric placement).
        """
        self._ema_has_updates = False
        if self._get_ema_callback(trainer) is None:
            self.map_metric_ema = None
            return
        if self.map_metric_ema is None:
            ema_iou_type: Any = ["bbox", "segm"] if self._use_segm_metrics else "bbox"
            self.map_metric_ema = MeanAveragePrecision(
                iou_type=ema_iou_type,
                class_metrics=True,
                max_detection_thresholds=[1, 10, self._max_dets],
                backend="faster_coco_eval",
                sync_on_compute=False,  # we merge state across ranks ourselves (see map_metric in setup)
            ).to(pl_module.device)
        else:
            self.map_metric_ema.reset()

    def _should_compute_ema(self, pl_module: Any) -> bool:
        """Decide — identically on every rank — whether to run the EMA metric ``compute()``.

        Under DDP, ``_merge_metric_state_across_ranks`` issues cross-rank collectives that every rank must
        participate in, or none may — a rank that skips desynchronises the NCCL collective sequence and deadlocks
        validation (#931 / #449).  Each rank votes ``1`` only when its EMA metric exists and received at least
        one batch update this epoch; a cross-rank ``all_reduce(MIN)`` makes the decision unanimous — a single
        rank voting 0 suppresses EMA compute on all ranks.

        Args:
            pl_module: The LightningModule (provides the device for the reduction).

        Returns:
            ``True`` iff every rank both holds an EMA metric object and received at least one batch update this
            epoch, making ``compute()`` safe to run identically on all ranks; ``False`` otherwise (EMA compute
            skipped uniformly on all ranks).
        """
        has_ema = self.map_metric_ema is not None and self._ema_has_updates
        vote = 1 if has_ema else 0
        if is_dist_avail_and_initialized():
            flag = torch.tensor([vote], device=getattr(pl_module, "device", "cpu"))
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            vote = int(flag.item())
        return bool(vote)

    # torchmetrics MeanAveragePrecision list-type state attributes — verified against torchmetrics >=1.2,<2
    # (pyproject.toml pin).  List states are identified by an empty-list default in metric._defaults.
    # On any torchmetrics upgrade, re-verify and update:
    #   python -c "from torchmetrics.detection import MeanAveragePrecision; \
    #              m = MeanAveragePrecision(); \
    #              print(sorted(k for k, v in m._defaults.items() if isinstance(v, list)))"
    # setup() asserts this tuple matches installed torchmetrics on every run.
    # TODO: remove this tuple (and the merge workaround) when Lightning-AI/torchmetrics#3199 is resolved.
    _MAP_STATE_ATTRS = (
        "detection_box",
        "detection_scores",
        "detection_labels",
        "detection_mask",
        "groundtruth_box",
        "groundtruth_labels",
        "groundtruth_mask",
        "groundtruth_crowds",
        "groundtruth_area",
    )

    def _merge_metric_state_across_ranks(self, metric: Any) -> None:
        """Merge a metric's accumulated per-rank state onto every rank, replacing torchmetrics' sync.

        torchmetrics' built-in sync (``gather_all_tensors``) varies the number of collectives by each state
        tensor's *local* ndim (scalar → 1 all_gather, vector → 2), so when DDP seg validation leaves a state
        scalar on some ranks and a vector on others the ranks issue different collective counts and deadlock
        (#931 / #449).  Instead we gather each state list once with the repo's pickle-based ``all_gather`` — a
        fixed collective pattern issued identically on every rank regardless of tensor shape — and concatenate.
        With ``sync_on_compute=False`` the metric's own ``compute()`` then runs locally over the merged full-set
        state, yielding the identical global mAP without any shape-dependent collective.

        Args:
            metric: The ``MeanAveragePrecision`` instance whose state should be merged in place.

        Note:
            No-op when ``metric`` is ``None``, when the distributed process group is not
            initialised, or when world size is 1 (single GPU / CPU training).  In these
            cases the metric state is unchanged.
        """
        if metric is None or not is_dist_avail_and_initialized() or get_world_size() == 1:
            return
        for attr in self._MAP_STATE_ATTRS:
            local = getattr(metric, attr, None)
            if local is None:
                continue
            # Move tensors to CPU so cross-device pickling during the gather is safe; RLE mask
            # entries are already CPU tuples and pass through unchanged.
            local_cpu = [v.detach().cpu() if torch.is_tensor(v) else v for v in local]
            gathered = all_gather(local_cpu)  # list of per-rank lists (identical on every rank)
            merged = [item for rank_list in gathered for item in rank_list]
            setattr(metric, attr, merged)
        # After merging, _update_count may still be 0 on ranks that received no local updates.
        # torchmetrics 1.x compute() works correctly regardless, but emits a UserWarning
        # ("compute called before update") that spams DDP logs on those ranks.
        metric._update_count = max(getattr(metric, "_update_count", 0), 1)

    @staticmethod
    def _metric_has_updates(metric: Any) -> bool:
        """Return whether a torchmetrics metric has accumulated at least one update."""
        update_count = getattr(metric, "_update_count", None)
        if isinstance(update_count, int):
            return update_count > 0
        if torch.is_tensor(update_count):
            return bool(update_count.detach().cpu().item() > 0)
        return True

    def _get_or_create_keypoint_oks_metric(self, trainer: Any, split: str) -> MetricKeypointOKS | None:
        """Return the :class:`~rfdetr.evaluation.keypoint_oks.MetricKeypointOKS` for *split*, creating it if needed.

        The metric is created lazily on first access per split and reused across epochs (state is reset
        at epoch boundaries via :meth:`_reset_keypoint_split`).

        Args:
            trainer: The PTL Trainer (provides access to the datamodule).
            split: One of ``"train"``, ``"val"``, ``"val_ema"``, or ``"test"``.

        Returns:
            A :class:`~rfdetr.evaluation.keypoint_oks.MetricKeypointOKS` bound to the split's COCO
            ground-truth, or ``None`` when no dataset is available.
        """
        if split in self._keypoint_oks_metrics:
            return self._keypoint_oks_metrics[split]

        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None:
            return None

        source_split = split.removesuffix("_ema")
        split_attrs = {
            "train": ("_dataset_train",),
            "val": ("_dataset_val",),
            "test": ("_dataset_test",),
        }.get(source_split, ("_dataset_val", "_dataset_test", "_dataset_train"))
        for attr in split_attrs:
            dataset = getattr(datamodule, attr, None)
            if dataset is None:
                continue
            coco_api = get_coco_api_from_dataset(dataset)
            if coco_api is None:
                continue
            metric = MetricKeypointOKS(
                coco_api,
                keypoint_oks_sigmas=self._keypoint_oks_sigmas,
                max_dets=self._max_dets,
            )
            self._keypoint_oks_metrics[split] = metric
            return metric
        return None

    def _reset_keypoint_split(self, split: str) -> None:
        """Reset accumulated keypoint predictions for *split*.

        Args:
            split: One of ``"train"``, ``"val"``, ``"val_ema"``, or ``"test"``.
        """
        metric = self._keypoint_oks_metrics.get(split)
        if metric is not None:
            metric.reset()

    def _update_keypoint_oks_metric(self, trainer: Any, outputs: dict[str, Any], split: str) -> None:
        """Accumulate batch predictions into the keypoint OKS metric.

        Args:
            trainer: The PTL Trainer.
            outputs: Batch output dict with ``"results"`` and ``"targets"`` keys.
            split: Metric split (``"train"``, ``"val"``, ``"val_ema"``, or ``"test"``).
        """
        if not self._keypoint_mode:
            return

        metric = self._get_or_create_keypoint_oks_metric(trainer, split)
        if metric is None:
            return

        predictions: dict[int, dict[str, torch.Tensor]] = {}
        results = outputs["results"]
        targets = outputs["targets"]
        for result, target in zip(results, targets):
            image_id_tensor = target.get("image_id")
            if image_id_tensor is None:
                continue
            image_id = int(image_id_tensor.item()) if torch.is_tensor(image_id_tensor) else int(image_id_tensor)
            if "keypoints" not in result:
                predictions[image_id] = {}
                continue
            predictions[image_id] = {
                "boxes": result["boxes"].detach().cpu(),
                "scores": result["scores"].detach().cpu(),
                "labels": result["labels"].detach().cpu(),
                "keypoints": result["keypoints"].detach().cpu(),
            }

        if not predictions:
            return
        metric.update(predictions)

    def _compute_and_log_keypoint_map(
        self,
        split: str,
        pl_module: Any,
        trainer: Any,
        *,
        log_split: str | None = None,
        metric_prefix: str = "",
    ) -> None:
        """Compute and log OKS keypoint AP/AR metrics when keypoint mode is active.

        Args:
            split: Internal metric split (``"val"``, ``"val_ema"``, ``"train"``, ``"test"``).
            pl_module: The LightningModule used to log scalar metrics.
            trainer: The PTL Trainer (provides ``callback_metrics``).
            log_split: Namespace prefix for logged keys. Defaults to *split*.
            metric_prefix: Optional string prepended to each metric name (e.g. ``"ema_"``).
        """
        metric = self._keypoint_oks_metrics.get(split)
        if not self._keypoint_mode or metric is None:
            return
        # Cross-rank vote before entering compute(): metric.compute() calls
        # synchronize_between_processes() which issues an all_gather collective. If any
        # rank short-circuits here without joining that collective the process group
        # deadlocks. Use the same all_reduce(MIN) pattern as _should_compute_ema.
        has_updates_vote = 1 if metric.has_updates else 0
        if is_dist_avail_and_initialized():
            flag = torch.tensor([has_updates_vote], device=getattr(pl_module, "device", "cpu"))
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            has_updates_vote = int(flag.item())
        if not has_updates_vote:
            return

        log_split = split if log_split is None else log_split
        try:
            stats = metric.compute()
            keypoint_metrics = {
                "keypoint_map_50_95": (OKSKey.MAP, True),
                "keypoint_map_50": (OKSKey.MAP_50, True),
                "keypoint_map_75": (OKSKey.MAP_75, False),
                "keypoint_mAR": (OKSKey.MAR, False),
            }
            for metric_name, (stat_key, prog_bar) in keypoint_metrics.items():
                value = stats.get(stat_key, -1.0)
                if value < 0:
                    continue
                log_key = f"{log_split}/{metric_prefix}{metric_name}"
                pl_module.log(log_key, value, prog_bar=prog_bar, logger=True, on_step=False, on_epoch=True)
                trainer.callback_metrics[log_key] = torch.tensor(value)
        finally:
            metric.reset()

    def _build_per_class_rows(
        self,
        metrics: dict[str, Any],
        pfx: str,
        split: str,
        pl_module: Any,
        ar_by_cid: dict[int, float],
        f1_by_cid: dict[int, dict[str, float]],
    ) -> list[dict[str, Any]]:
        """Build per-class rows and emit per-class AP metrics.

        Args:
            metrics: Output of ``MeanAveragePrecision.compute()``.
            pfx: Key prefix for bbox metrics when segmentation mode is enabled.
            split: Metric namespace (``"val"`` or ``"test"``).
            pl_module: LightningModule used for metric logging.
            ar_by_cid: Per-class AR keyed by ``category_id``.
            f1_by_cid: Per-class F1/precision/recall keyed by ``category_id``.

        Returns:
            Per-class rows for table rendering.
        """
        per_class: list[dict[str, Any]] = []
        if not self._log_per_class_metrics:
            return per_class

        pc_key = f"{pfx}map_per_class"
        if pc_key not in metrics or "classes" not in metrics:
            return per_class

        for class_id, ap in zip(metrics["classes"], metrics[pc_key]):
            ap_f = float(ap)
            ar_f = ar_by_cid.get(int(class_id), float("nan"))
            if ap_f < 0 and (ar_f != ar_f or ar_f < 0):  # no ground-truth: skip ghost class
                continue
            idx = int(class_id)
            name = self._cat_id_to_name.get(idx, str(idx))
            pl_module.log(f"{split}/AP/{name}", ap)
            row: dict[str, Any] = {"name": name, "ap": ap_f, "ar": ar_f}
            row.update(f1_by_cid.get(idx, {"f1": float("nan"), "precision": float("nan"), "recall": float("nan")}))
            per_class.append(row)
        return per_class

    def _print_metrics_tables(
        self,
        trainer: Any,
        split: str,
        overall: dict[str, float],
        per_class: list[dict[str, Any]],
    ) -> None:
        """Print two tables to the terminal: overall metrics and per-class metrics.

        The overall table is transposed (metrics as columns, one value row) with true merged group-header cells rendered
        via box-drawing characters: ``mAP`` spans sub-columns 50:95 / 50 / 75, ``mAR`` spans ``@N``, and ``F1 sweep``
        spans F1 / Prec / Recall.  The per-class table uses a standard Rich ``Table`` with columns for AP 50:95, AR, F1,
        Prec, Recall.

        Only runs on the global-zero rank to avoid duplicate output in DDP.

        Args:
            trainer: The PTL Trainer (used to check ``is_global_zero``).
            split: ``"val"`` or ``"test"``.
            overall: Ordered mapping of metric label → scalar value.
            per_class: Per-class dicts with keys ``name``, ``ap``, ``ar``,
                ``f1``, ``precision``, ``recall``; skipped when empty.
        """
        if not getattr(trainer, "is_global_zero", True):
            return
        if not _IS_RICH_AVAILABLE:
            self._missing_rich_warning_emitted = _warn_missing_rich_once(self._missing_rich_warning_emitted)
            return

        console = _get_rich_console(trainer)
        current_epoch = int(getattr(trainer, "current_epoch", 0)) + 1
        max_epochs = getattr(trainer, "max_epochs", None)
        epoch_sfx = (
            f" (Epoch {current_epoch}/{max_epochs})"
            if isinstance(max_epochs, int) and max_epochs > 0
            else f" (Epoch {current_epoch})"
        )
        title_pfx = split.capitalize() + epoch_sfx
        overall_rendered = _render_overall_merged(title_pfx, overall, self._max_dets)

        if self._in_notebook:
            # Lazily create an ipywidgets.Output on the first table print so it
            # anchors below the progress bar that is already visible.  Subsequent
            # epochs clear only the widget's isolated slot — the main cell output
            # (and PTL's progress bar) is never touched, so there is no flicker.
            if self._output_widget is None:
                with contextlib.suppress(ImportError):
                    import ipywidgets as widgets
                    from IPython.display import display

                    self._output_widget = widgets.Output()
                    display(self._output_widget)

            if self._output_widget is not None:
                self._output_widget.clear_output(wait=True)
                with self._output_widget:
                    _render_summary_tables(console, title_pfx, overall_rendered, per_class)
                return

            # ipywidgets not installed — fall back to IPython cell-level clear so
            # tables replace each other instead of accumulating across epochs.
            with contextlib.suppress(ImportError):
                from IPython.display import clear_output

                clear_output(wait=True)
            _render_summary_tables(console, title_pfx, overall_rendered, per_class)
            return

        # Print directly through the console.  A second rich.live.Live on the same
        # console as RichProgressBar would silently nest (Live._nested=True) and
        # delegate all refresh() calls to the progress-bar renderable, so metric
        # tables would never appear.  console.print() avoids that nesting issue.
        _render_summary_tables(console, title_pfx, overall_rendered, per_class)

    def _convert_preds(self, preds: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        """Normalise prediction dicts from ``PostProcess`` for torchmetrics.

        ``PostProcess.forward`` returns masks with shape ``[K, 1, H, W]`` (the extra channel is introduced by
        ``F.interpolate`` which requires 4-D input).  Both ``torchmetrics.MeanAveragePrecision`` and
        ``engine.build_matching_data`` expect ``[K, H, W]``, so squeeze the channel dim when present.

        ``PostProcess.forward`` currently returns ``[K, 1, H, W]`` masks. Keep this callback-local squeeze for metric
        code paths because ``RFDETR.predict`` and other inference-facing callers still consume the 4-D representation
        and apply ``.squeeze(1)`` at their boundary.

        Args:
            preds: Raw per-image prediction dicts from ``PostProcess``.

        Returns:
            Per-image dicts with ``masks`` squeezed to ``[K, H, W]`` when applicable; all other keys are passed through
            unchanged.
        """
        out = []
        for p in preds:
            entry = dict(p)
            if "masks" in entry and entry["masks"].ndim == 4 and entry["masks"].shape[1] == 1:
                entry["masks"] = entry["masks"].squeeze(1)
            out.append(entry)
        return out

    def _convert_targets(self, targets: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        """Convert targets from normalised CxCyWH to absolute xyxy boxes.

        Also passes ``iscrowd`` and ``masks`` through unchanged.

        Args:
            targets: Per-image target dicts with ``boxes`` in normalised
                CxCyWH format and ``orig_size`` as ``[H, W]``.

        Returns:
            Per-image dicts with ``boxes`` in absolute xyxy, ``labels``, and optionally ``masks`` and ``iscrowd``.
        """
        out = []
        for t in targets:
            h, w = t["orig_size"].tolist()
            scale = t["boxes"].new_tensor([w, h, w, h])
            boxes = box_cxcywh_to_xyxy(t["boxes"]) * scale
            entry: dict[str, torch.Tensor] = {"boxes": boxes, "labels": t["labels"]}
            if "masks" in t:
                masks = t["masks"].bool()
                # PostProcess resizes predicted masks to orig_size; resize GT
                # masks to match so that mask-IoU comparisons are size-consistent.
                if masks.shape[-2:] != (int(h), int(w)):
                    masks = (
                        F.interpolate(
                            masks.float().unsqueeze(1),
                            size=(int(h), int(w)),
                            mode="nearest",
                        )
                        .squeeze(1)
                        .bool()
                    )
                entry["masks"] = masks
            if "iscrowd" in t:
                entry["iscrowd"] = t["iscrowd"]
            out.append(entry)
        return out
