# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Best-model checkpointing and early stopping callbacks for RF-DETR Lightning training."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning import __version__ as ptl_version
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from mmdet.models.layers.rf_detr.training.callbacks.ema import RFDETREMACallback
from mmdet.models.layers.rf_detr.utilities.logger import get_logger
from mmdet.models.layers.rf_detr.utilities.package import get_version
from mmdet.models.layers.rf_detr.utilities.state_dict import _make_fit_loop_state, strip_checkpoint

logger = get_logger()


class BestModelCallback(ModelCheckpoint):
    """Track best validation mAP and save best checkpoints during training.

    Extends :class:`pytorch_lightning.callbacks.ModelCheckpoint` to save stripped ``{model, args, epoch}`` ``.pth``
    files (instead of full ``.ckpt`` files) and to track a separate EMA checkpoint in parallel.

    At the end of training the overall winner (regular vs EMA, strict ``>`` for EMA) is copied to
    ``checkpoint_best_total.pth`` and optimizer/scheduler state is stripped via
    :func:`rfdetr.util.misc.strip_checkpoint`.

    Checkpoints are only updated on validation epochs where the monitor metric is actually logged.  On non-eval epochs
    (when ``eval_interval > 1`` causes COCO evaluation to be skipped) the callback is a no-op.

    ``state_dict()`` and ``load_state_dict()`` are overridden to persist ``_best_ema`` in the Lightning callback state,
    ensuring that ``trainer.fit(ckpt_path=...)`` resumes EMA high-water-mark tracking from the correct value.

    Args:
        output_dir: Directory where checkpoint files are written.
        monitor_regular: Metric key for the regular model mAP.
        monitor_ema: Metric key for the EMA model mAP.  ``None`` disables EMA tracking.
        run_test: If ``True``, run ``trainer.test()`` on the best model at the end of training.
        skip_best_epochs: Ignore the first N epochs (0..N-1) when tracking
            best regular and EMA checkpoints.  Useful when fine-tuning from ``pretrain_weights``: the pretrained model's
            epoch-0 mAP can artificially dominate best-checkpoint selection before training adapts to the new dataset.
        smooth_alpha: Exponential-moving-average smoothing factor in ``[0.0, 1.0)`` applied to the regular monitor
            metric before checkpoint comparison.  ``0.0`` (default) disables smoothing and keeps legacy behaviour:
            ``trainer.callback_metrics[monitor_regular]`` is consumed as-is by the parent
            :class:`~pytorch_lightning.callbacks.ModelCheckpoint`.  ``smooth_alpha > 0`` maintains an internal EMA
            state ``self._smoothed_regular = alpha * self._smoothed_regular + (1 - alpha) * raw`` and temporarily
            substitutes the smoothed value into ``trainer.callback_metrics`` for the duration of the parent's
            improvement check; the original raw value is always restored before returning so what gets logged to
            ``metrics.csv`` is unaffected.  Useful for noisy metrics (e.g. keypoint mAP under NLL-Cholesky losses) where
            raw per-epoch swings can lock the best checkpoint to an early local peak.  The EMA accumulator is updated
            on every validation epoch including epochs within the ``skip_best_epochs`` window so the smoothed value
            is pre-warmed by the first eligible comparison.

    Examples:
        Skip the first 3 epochs so pretrained weights do not dominate:

        >>> import tempfile
        >>> from mmdet.models.layers.rf_detr.training.callbacks.best_model import BestModelCallback
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     cb = BestModelCallback(output_dir=tmp, skip_best_epochs=3)
        ...     cb._skip_best_epochs
        3
    """

    FILE_EXTENSION = ".pth"

    def __init__(
        self,
        output_dir: str,
        monitor_regular: str = "val/mAP_50_95",
        monitor_ema: str | None = None,
        run_test: bool = True,
        skip_best_epochs: int = 0,
        smooth_alpha: float = 0.0,
    ) -> None:
        super().__init__(
            dirpath=output_dir,
            filename="checkpoint_best_regular",
            monitor=monitor_regular,
            mode="max",
            save_top_k=1,
            save_on_train_epoch_end=False,
            verbose=False,
            auto_insert_metric_name=False,
            enable_version_counter=False,
        )
        self._monitor_ema = monitor_ema
        self._run_test = run_test
        self._best_ema: float = 0.0
        self._output_dir = Path(output_dir)
        if isinstance(skip_best_epochs, bool) or not isinstance(skip_best_epochs, int):
            raise TypeError("skip_best_epochs must be a non-negative integer")
        if skip_best_epochs < 0:
            raise ValueError("skip_best_epochs must be greater than or equal to 0")
        self._skip_best_epochs = skip_best_epochs
        if isinstance(smooth_alpha, bool) or not isinstance(smooth_alpha, (int, float)):
            raise TypeError("smooth_alpha must be a float in [0.0, 1.0)")
        if not math.isfinite(smooth_alpha) or smooth_alpha < 0.0 or smooth_alpha >= 1.0:
            raise ValueError("smooth_alpha must be in [0.0, 1.0)")
        self._smooth_alpha: float = float(smooth_alpha)
        # EMA accumulator for the regular monitor metric.  Only updated when
        # ``smooth_alpha > 0``; persisted via state_dict()/load_state_dict() so resumed
        # training continues smoothing from the correct value rather than restarting at 0.0.
        self._smoothed_regular: float = 0.0
        # Best raw (non-smoothed) regular metric value at the most recent epoch where the
        # parent saved a new best checkpoint.  Used in on_fit_end to compare raw EMA vs raw
        # regular — best_model_score tracks the smoothed value when smooth_alpha > 0.
        self._best_raw_regular: float = 0.0
        # Stash current pl_module so _save_checkpoint (no pl_module param) can access it.
        self._current_pl_module: LightningModule | None = None

    @staticmethod
    def _build_checkpoint_payload(
        model_state_dict: dict[str, torch.Tensor],
        args_dict: object,
        trainer: Trainer,
        model_name: str | None = None,
        model_config_dict: object | None = None,
    ) -> dict[str, object]:
        """Build a PTL-compatible RF-DETR checkpoint payload.

        Args:
            model_state_dict: Model weights with raw (non-prefixed) keys.
            args_dict: Serialized training args/config payload.
            trainer: Active Lightning trainer providing epoch/step counters.
            model_name: Name of the model class (e.g. ``"RFDETRLarge"``).
            model_config_dict: Serialized architecture config needed to reconstruct schema-dependent models.

        Returns:
            Checkpoint dictionary that supports ``Trainer.fit(ckpt_path=...)`` while intentionally omitting
            optimizer/scheduler states.
        """
        payload: dict[str, object] = {
            "model": model_state_dict,
            "args": args_dict,
            "epoch": trainer.current_epoch,
            # PTL-compatible keys so trainer.fit(ckpt_path=...) works directly.
            "state_dict": {f"model.{k}": v for k, v in model_state_dict.items()},
            "global_step": trainer.global_step,
            "pytorch-lightning_version": ptl_version,
            "loops": {
                "fit_loop": _make_fit_loop_state(trainer.current_epoch),
                # Minimal stubs so trainer.validate(ckpt_path=...) and trainer.test(ckpt_path=...)
                # don't crash on KeyError — PTL restore_loops() expects both keys (PTL>=2.x,
                # checkpoint_connector.py:restore_loops).
                "validate_loop": {"state_dict": {}},
                "test_loop": {"state_dict": {}},
            },
            # Keep keys present with empty values so PTL resume paths that
            # expect them can proceed without loading optimizer state.
            "optimizer_states": [],
            "lr_schedulers": [],
        }
        # Only write model_name when resolved — omit the key entirely when None
        # so old-format and unresolved checkpoints are indistinguishable.
        if model_name is not None:
            payload["model_name"] = model_name
        if model_config_dict is not None:
            payload["model_config"] = model_config_dict
        # Record the rfdetr package version for provenance / compatibility hints.
        # Omit the key when the version cannot be resolved (e.g. editable install
        # without package metadata) so old-format checkpoints are indistinguishable.
        version = get_version()
        if version is not None:
            payload["rfdetr_version"] = version
        return payload

    @staticmethod
    def _get_live_model_state_dict(pl_module: LightningModule) -> dict[str, torch.Tensor]:
        """Resolve live model weights from the active Lightning module.

        Args:
            pl_module: The ``RFDETRModelModule`` being trained.

        Returns:
            State dict from the live, non-EMA model, unwrapped from ``torch.compile`` when needed.
        """
        _orig = getattr(pl_module.model, "_orig_mod", None)
        raw = _orig if isinstance(_orig, torch.nn.Module) else pl_module.model
        return raw.state_dict()

    @staticmethod
    def _get_ema_model_state_dict(
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> dict[str, torch.Tensor]:
        """Resolve EMA model weights from the active EMA callback.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``RFDETRModelModule`` being trained.

        Returns:
            EMA model state dict when available, otherwise the live model state dict.
        """
        for callback in trainer.callbacks:
            getter = getattr(callback, "get_ema_model_state_dict", None)
            if callable(getter):
                state_dict = getter()
                if state_dict is not None:
                    return state_dict
                break
        logger.warning(
            "EMA metric improved but EMA callback weights were unavailable; saving current model weights as fallback."
        )
        _orig = getattr(pl_module.model, "_orig_mod", None)
        raw = _orig if isinstance(_orig, torch.nn.Module) else pl_module.model
        return raw.state_dict()

    @staticmethod
    def _serialize_model_config(
        pl_module: LightningModule, state_dict: dict[str, Any] | None = None
    ) -> dict[str, object] | None:
        """Serialize the model architecture config when the module exposes one.

        Schema-critical fields (``num_keypoints_per_class``, ``num_classes``) are synced from live model weights so the
        saved config reflects what the model actually learned, not a stale constructor default (e.g. COCO ``[0, 17]``
        when the model was fine-tuned on ``[0, 33]``).

        Args:
            pl_module: The Lightning module whose model config will be serialized.
            state_dict: Pre-computed model state dict. If ``None``, computed from the live model.
                Pass the already-retrieved checkpoint state dict to avoid a redundant copy.
        """
        model_config = getattr(pl_module, "model_config", None)
        if model_config is None:
            return None
        if isinstance(model_config, dict):
            return model_config
        model_dump = getattr(model_config, "model_dump", None)
        if not callable(model_dump):
            return None
        dumped = model_dump()
        if not isinstance(dumped, dict):
            return None

        # Sync schema-critical fields from live model weights.
        if state_dict is None:
            _orig = getattr(pl_module.model, "_orig_mod", None)
            raw = _orig if isinstance(_orig, torch.nn.Module) else pl_module.model
            state_dict = raw.state_dict()

        _kp_mask = state_dict.get("_kp_active_mask")
        if isinstance(_kp_mask, torch.Tensor) and _kp_mask.ndim == 2 and "num_keypoints_per_class" in dumped:
            dumped["num_keypoints_per_class"] = [int(n) for n in _kp_mask.sum(dim=1).tolist()]

        _ce_weight = state_dict.get("class_embed.weight")
        if isinstance(_ce_weight, torch.Tensor) and _ce_weight.ndim == 2 and "num_classes" in dumped:
            dumped["num_classes"] = _ce_weight.shape[0] - 1  # shape[0] = num_classes + 1 (background)

        return dumped

    @staticmethod
    def _resolve_model_name(pl_module: LightningModule) -> str | None:
        """Resolve checkpoint model_name from model_config or config type.

        The CLI/PTL path does not call ``RFDETR.train()``, so ``model_config.model_name`` may be unset. In that case,
        infer the model class from concrete config names like ``RFDETRSmallConfig``.

        Note:
            The ``DeprecatedConfig`` ``RuntimeError`` guard is only reachable from the CLI/PTL path. ``RFDETR.train()``
            pre-populates ``model_config.model_name`` before saving any checkpoint, so the config type-name branch (and
            therefore the ``DeprecatedConfig`` guard) is never reached when training is started via ``RFDETR.train()``.
        """
        model_config = getattr(pl_module, "model_config", None)
        configured_name = getattr(model_config, "model_name", None) if model_config is not None else None
        if isinstance(configured_name, str):
            normalized_name = configured_name.strip()
            if normalized_name:
                return normalized_name

        config_type_name = type(model_config).__name__ if model_config is not None else ""

        if config_type_name.endswith("DeprecatedConfig"):
            raise RuntimeError(
                f"Deprecated model config '{config_type_name}' is no longer supported. "
                "Re-train your model using a current model variant."
            )
        if config_type_name.startswith("RFDETR") and config_type_name.endswith("Config"):
            return config_type_name.removesuffix("Config")
        return None

    def state_dict(self) -> dict[str, Any]:
        """Return callback state including ``_best_ema``, ``_smoothed_regular``, and ``_best_raw_regular``.

        Extends the parent :class:`~pytorch_lightning.callbacks.ModelCheckpoint` state dict with three extra keys so
        that ``trainer.fit(ckpt_path=...)`` resumes the EMA high-water mark, the smoothed-metric accumulator, and the
        raw metric at the smoothed-best epoch from their correct values rather than resetting to ``0.0``.

        Returns:
            State dict with all parent fields plus ``"_best_ema"``, ``"_smoothed_regular"``,
            and ``"_best_raw_regular"``.
        """
        state = super().state_dict()
        state["_best_ema"] = self._best_ema
        state["_smoothed_regular"] = self._smoothed_regular
        state["_best_raw_regular"] = self._best_raw_regular
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore callback state from a Lightning checkpoint.

        Pops ``"_best_ema"``, ``"_smoothed_regular"``, and ``"_best_raw_regular"`` from a shallow copy of *state_dict*
        before delegating to the parent so the parent does not receive unexpected keys.  Each key defaults to ``0.0``
        when absent (e.g. checkpoints saved before these fields were persisted) and is reset to ``0.0`` when the stored
        value is non-finite.

        Args:
            state_dict: Callback state dict as produced by :meth:`state_dict`.
        """
        # Copy to avoid mutating the caller's dict — PTL may reuse it.
        state = dict(state_dict)
        self._best_ema = float(state.pop("_best_ema", 0.0))
        if not math.isfinite(self._best_ema):
            self._best_ema = 0.0
        self._smoothed_regular = float(state.pop("_smoothed_regular", 0.0))
        if not math.isfinite(self._smoothed_regular):
            self._smoothed_regular = 0.0
        self._best_raw_regular = float(state.pop("_best_raw_regular", 0.0))
        if not math.isfinite(self._best_raw_regular):
            self._best_raw_regular = 0.0
        super().load_state_dict(state)

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        """Save stripped ``.pth`` format instead of a full ``.ckpt``.

        Skips on non-main processes.  Intentionally does NOT call ``trainer.save_checkpoint()`` — we only want ``{model,
        args, epoch}``.

        Args:
            trainer: The Lightning Trainer instance.
            filepath: Destination path (ends in ``.pth`` via ``FILE_EXTENSION``).
        """
        if not trainer.is_global_zero:
            return
        pl_module = self._current_pl_module
        if pl_module is None:
            raise RuntimeError(
                f"BestModelCallback._save_checkpoint called with filepath={filepath!r} "
                f"at epoch={trainer.current_epoch} but pl_module was not set."
            )
        pth_path = Path(filepath)
        pth_path.parent.mkdir(parents=True, exist_ok=True)
        # Regular metrics are produced from the live validation step.  Keep the regular
        # checkpoint tied to live weights even when an EMA checkpoint is tracked in
        # parallel; checkpoint_best_ema.pth is the only file that should save EMA weights.
        model_state_dict = self._get_live_model_state_dict(pl_module)
        # Enrich train_config with dataset class names so reloaded checkpoints
        # return the correct labels, not COCO defaults (#509).
        train_config = pl_module.train_config
        dataset_class_names = getattr(trainer.datamodule, "class_names", None)
        if (
            dataset_class_names is not None
            and hasattr(train_config, "model_copy")
            and getattr(train_config, "class_names", None) is None
        ):
            train_config = train_config.model_copy(update={"class_names": dataset_class_names})
        args_dict = train_config.model_dump() if hasattr(train_config, "model_dump") else train_config
        model_name = self._resolve_model_name(pl_module)
        model_config_dict = self._serialize_model_config(pl_module, model_state_dict)
        torch.save(
            self._build_checkpoint_payload(
                model_state_dict,
                args_dict,
                trainer,
                model_name=model_name,
                model_config_dict=model_config_dict,
            ),
            pth_path,
        )
        self._last_global_step_saved = trainer.global_step
        monitor_value = trainer.callback_metrics.get(self.monitor)
        if torch.is_tensor(monitor_value):
            monitor_display = f"{monitor_value.item():.6g}"
        elif monitor_value is not None:
            monitor_display = str(monitor_value)
        else:
            monitor_display = "unknown"
        logger.info(
            "Best regular checkpoint saved to %s (epoch %d, monitor=%s, value=%s)",
            pth_path,
            trainer.current_epoch,
            self.monitor,
            monitor_display,
        )

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save best regular/EMA checkpoints when validation mAP improves.

        Delegates regular-model checkpoint management to the :class:`~pytorch_lightning.callbacks.ModelCheckpoint`
        parent (handles improvement detection, fast_dev_run/sanity guards, ``best_model_path`` and ``best_model_score``
        bookkeeping).  EMA is tracked independently.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``RFDETRModelModule`` being trained.
        """
        # Stash before the skip guard — eligible epochs still need this reference
        # inside _save_checkpoint (which receives no pl_module param).
        self._current_pl_module = pl_module
        # Update the EMA accumulator before the skip guard so it warms up during skipped
        # epochs and avoids cold-start deflation at the first eligible epoch.
        raw: float | None = None
        if self._smooth_alpha > 0.0 and self.monitor in trainer.callback_metrics:
            raw = trainer.callback_metrics[self.monitor].item()
            self._smoothed_regular = self._smooth_alpha * self._smoothed_regular + (1.0 - self._smooth_alpha) * raw
        if trainer.current_epoch < self._skip_best_epochs:
            return
        # Guard: only run checkpoint logic when the monitored metric was actually
        # logged this epoch (non-eval epochs with eval_interval > 1 skip COCO eval
        # so the key is absent from callback_metrics).
        if self.monitor not in trainer.callback_metrics:
            return
        # Optional EMA smoothing of the monitored metric before the parent's improvement
        # check.  The smoothed value is substituted into ``trainer.callback_metrics`` for
        # the duration of the super() call only; the original raw tensor is always
        # restored in the ``finally`` block so what gets logged to ``metrics.csv`` and
        # seen by other callbacks (including EMA tracking below) is unaffected.
        if self._smooth_alpha > 0.0:
            # raw was captured in the accumulator update above; smooth_alpha > 0 and monitor
            # in callback_metrics are both guaranteed here (passed the guard above).
            current_raw: float = raw if raw is not None else trainer.callback_metrics[self.monitor].item()
            prev_best_score = self.best_model_score.item() if self.best_model_score is not None else -float("inf")
            original = trainer.callback_metrics[self.monitor]
            trainer.callback_metrics[self.monitor] = torch.tensor(
                self._smoothed_regular, dtype=original.dtype, device=original.device
            )
            try:
                super().on_validation_end(trainer, pl_module)
            finally:
                trainer.callback_metrics[self.monitor] = original
            # Record the raw metric value when parent selected a new best so on_fit_end
            # compares raw EMA vs raw regular (not raw EMA vs smoothed regular).
            new_best_score = self.best_model_score.item() if self.best_model_score is not None else -float("inf")
            if new_best_score > prev_best_score:
                self._best_raw_regular = current_raw
        else:
            super().on_validation_end(trainer, pl_module)

        # EMA model — custom tracking on top of parent.
        if self._monitor_ema is None or not trainer.is_global_zero:
            return
        ema_val = trainer.callback_metrics.get(self._monitor_ema, torch.tensor(0.0)).item()
        if ema_val > self._best_ema:
            self._best_ema = ema_val
            self._output_dir.mkdir(parents=True, exist_ok=True)
            ema_state_dict = self._get_ema_model_state_dict(trainer, pl_module)
            # Enrich train_config with dataset class names so reloaded checkpoints
            # return the correct labels, not COCO defaults (#509).
            ema_train_config = pl_module.train_config
            dataset_class_names = getattr(trainer.datamodule, "class_names", None)
            if (
                dataset_class_names is not None
                and hasattr(ema_train_config, "model_copy")
                and getattr(ema_train_config, "class_names", None) is None
            ):
                ema_train_config = ema_train_config.model_copy(update={"class_names": dataset_class_names})
            ema_args_dict = (
                ema_train_config.model_dump() if hasattr(ema_train_config, "model_dump") else ema_train_config
            )
            ema_model_name = self._resolve_model_name(pl_module)
            ema_model_config_dict = self._serialize_model_config(pl_module, ema_state_dict)
            torch.save(
                self._build_checkpoint_payload(
                    ema_state_dict,
                    ema_args_dict,
                    trainer,
                    model_name=ema_model_name,
                    model_config_dict=ema_model_config_dict,
                ),
                self._output_dir / "checkpoint_best_ema.pth",
            )
            logger.info(
                "Best EMA mAP improved to %.4f (epoch %d)",
                ema_val,
                trainer.current_epoch,
            )

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Select the overall best model and optionally run test evaluation.

        Copies the winner (regular vs EMA, strict ``>`` for EMA) to ``checkpoint_best_total.pth``, strips
        optimizer/scheduler state, then optionally runs ``trainer.test()``.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``RFDETRModelModule`` being trained.
        """
        if not trainer.is_global_zero:
            return

        # Use _best_raw_regular when smoothing is active: best_model_score tracks the
        # smoothed value, so comparing it directly against raw _best_ema is biased.
        # _best_raw_regular is the raw (un-smoothed) metric value at the epoch where the
        # SMOOTHED metric last set a new high-water mark — i.e. the raw value for the
        # checkpoint that was actually saved.  This is intentionally not the all-time raw
        # peak: the saved checkpoint corresponds to the smoothed-best epoch, which may differ
        # from the epoch with the highest raw reading.  Do not change this to an all-time raw
        # max without adding a corresponding saved checkpoint for that epoch.
        if self._smooth_alpha > 0.0:
            best_regular = self._best_raw_regular
        else:
            best_regular = self.best_model_score.item() if self.best_model_score is not None else 0.0
        regular_path = Path(self.best_model_path) if self.best_model_path else None
        ema_path = self._output_dir / "checkpoint_best_ema.pth"
        total_path = self._output_dir / "checkpoint_best_total.pth"

        # Strict > for EMA to win (matches legacy behaviour).
        best_is_ema = self._best_ema > best_regular
        best_path = ema_path if (best_is_ema and ema_path.exists()) else regular_path

        if best_path and best_path.exists():
            shutil.copy2(best_path, total_path)
            strip_checkpoint(total_path)
            logger.info(
                "Best total checkpoint saved from %s (regular=%.4f, ema=%.4f)",
                "EMA" if best_is_ema else "regular",
                best_regular,
                self._best_ema,
            )

        if self._run_test:
            # Only call trainer.test() when the module actually defines test_step().
            cls_test_step = getattr(type(pl_module), "test_step", None)
            has_test_step = cls_test_step is not None and cls_test_step is not LightningModule.test_step
            if has_test_step:
                if not total_path.exists():
                    logger.warning(
                        "Skipping trainer.test() because no best checkpoint was produced. "
                        "Ensure the monitored metric is logged on evaluation epochs, that evaluation "
                        "runs often enough, and that skip_best_epochs is smaller than the number of "
                        "training epochs."
                    )
                    return
                # Load best weights before test — mirrors legacy main.py:602-609.
                # trust=True: checkpoint_best_total.pth is produced locally; allow pickle fallback if needed.
                from mmdet.models.layers.rf_detr.util.io import _safe_torch_load

                ckpt = _safe_torch_load(total_path, trust=True)
                # Checkpoints always store plain keys; load into the unwrapped module
                # so compiled (OptimizedModule) and non-compiled models both work.
                _orig = getattr(pl_module.model, "_orig_mod", None)
                raw = _orig if isinstance(_orig, torch.nn.Module) else pl_module.model
                raw.load_state_dict(ckpt["model"], strict=True)
                logger.info("Loaded best weights from %s for test evaluation.", total_path)
                # The EMA callback swaps final-EMA weights in for test epochs, which would silently overwrite the
                # just-loaded best weights — suppress its swap for this run only, restoring the default afterwards
                # so standalone trainer.test() calls keep evaluating EMA weights.
                ema_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, RFDETREMACallback)]
                prior_flags = [cb.suppress_test_swap for cb in ema_callbacks]
                for ema_callback in ema_callbacks:
                    ema_callback.suppress_test_swap = True
                try:
                    trainer.test(pl_module, datamodule=trainer.datamodule, verbose=False)
                finally:
                    for ema_callback, prior in zip(ema_callbacks, prior_flags):
                        ema_callback.suppress_test_swap = prior


class RFDETREarlyStopping(EarlyStopping):
    """Early stopping callback monitoring validation mAP for RF-DETR.

    Extends :class:`pytorch_lightning.callbacks.EarlyStopping` with dual-metric
    monitoring: by default it monitors ``max(regular_mAP, ema_mAP)`` (legacy
    behaviour); set ``use_ema=True`` to monitor the EMA metric exclusively.

    The effective metric is injected into ``trainer.callback_metrics`` under a synthetic key before delegating to the
    parent's stopping logic, so all parent features are available for free: ``state_dict``/``load_state_dict`` for
    checkpoint resumption, NaN/inf guard via ``check_finite``, and ``stopping_threshold``/``divergence_threshold``.

    Early stopping evaluates only on validation epochs where the monitored metrics are logged; non-eval epochs
    (``eval_interval > 1``) are skipped automatically.

    Args:
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum mAP improvement to reset the patience counter.
        use_ema: When ``True`` and both regular and EMA metrics are available,
            monitor only the EMA metric.  When ``False``, monitor ``max(regular, ema)``.
        monitor_regular: Metric key for the regular model mAP.
        monitor_ema: Metric key for the EMA model mAP.
        verbose: If ``True``, log early stopping status each epoch.
        skip_best_epochs: Ignore the first N epochs (0..N-1) when evaluating
            patience and best-score baselines.  Set this when fine-tuning from ``pretrain_weights`` to avoid premature
            stopping before the model adapts to the new dataset.

    Examples:
        Fine-tuning from pretrained weights — skip first 3 epochs:

        >>> from mmdet.models.layers.rf_detr.training.callbacks.best_model import RFDETREarlyStopping
        >>> cb = RFDETREarlyStopping(patience=10, skip_best_epochs=3)
        >>> cb._skip_best_epochs
        3
    """

    _SYNTHETIC_MONITOR: str = "__rfdetr_effective_map__"

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        use_ema: bool = False,
        monitor_regular: str = "val/mAP_50_95",
        monitor_ema: str = "val/ema_mAP_50_95",
        verbose: bool = True,
        skip_best_epochs: int = 0,
    ) -> None:
        super().__init__(
            monitor=self._SYNTHETIC_MONITOR,
            mode="max",
            patience=patience,
            min_delta=min_delta,
            check_on_train_epoch_end=False,
            verbose=verbose,
            check_finite=True,
            strict=False,  # We inject the key ourselves; don't crash if temporarily absent.
            log_rank_zero_only=True,
        )
        if isinstance(skip_best_epochs, bool) or not isinstance(skip_best_epochs, int):
            raise TypeError("skip_best_epochs must be a non-negative integer")
        if skip_best_epochs < 0:
            raise ValueError("skip_best_epochs must be greater than or equal to 0")

        self._monitor_regular = monitor_regular
        self._monitor_ema = monitor_ema
        self._use_ema = use_ema
        self._skip_best_epochs = skip_best_epochs

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Compute effective mAP and delegate to parent stopping logic.

        Computes ``ema_mAP`` or ``max(regular_mAP, ema_mAP)`` depending on ``use_ema``, injects the result under the
        synthetic monitor key, then calls :meth:`EarlyStopping.on_validation_end` which handles patience,
        ``trainer.should_stop``, logging, and ``state_dict`` persistence.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``RFDETRModelModule`` being trained.
        """
        if trainer.current_epoch < self._skip_best_epochs:
            return

        metrics = trainer.callback_metrics
        regular_tensor = metrics.get(self._monitor_regular)
        ema_tensor = metrics.get(self._monitor_ema)

        regular_val: float | None = regular_tensor.item() if regular_tensor is not None else None
        ema_val: float | None = ema_tensor.item() if ema_tensor is not None else None

        if regular_val is None and ema_val is None:
            return  # No metrics available — skip (matches legacy noop behaviour).

        if self._use_ema and ema_val is not None:
            effective = ema_val
        elif regular_val is not None and ema_val is not None:
            effective = max(regular_val, ema_val)
        elif ema_val is not None:
            effective = ema_val
        else:
            effective = regular_val  # type: ignore[assignment]

        trainer.callback_metrics[self._SYNTHETIC_MONITOR] = torch.tensor(effective)
        super().on_validation_end(trainer, pl_module)
