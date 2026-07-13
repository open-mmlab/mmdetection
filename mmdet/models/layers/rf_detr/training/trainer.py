# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Trainer factory — assembles a PTL Trainer from RF-DETR configs."""

import warnings
from typing import Any

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, TQDMProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy as _DDPStrategy

# _MultiProcessingLauncher is a private PTL API (leading underscore) that may change
# in minor PTL releases within the >=2.6,<3 range.  No public equivalent exists in
# PTL 2.x.  Monitor PTL changelogs when bumping the lower bound.
try:
    from pytorch_lightning.strategies.launchers.multiprocessing import _MultiProcessingLauncher
except ImportError:  # pragma: no cover - exercised in unit tests via monkeypatch
    _MultiProcessingLauncher = None  # type: ignore[assignment]

from mmdet.models.layers.rf_detr.config import KeypointTrainConfig, ModelConfig, TrainConfig
from mmdet.models.layers.rf_detr.training.callbacks import (
    BestModelCallback,
    DropPathCallback,
    RFDETREarlyStopping,
    RFDETREMACallback,
)
from mmdet.models.layers.rf_detr.training.callbacks.coco_eval import COCOEvalCallback
from mmdet.models.layers.rf_detr.utilities.logger import get_logger

_logger = get_logger()


def _try_import_tensorboard_summary_writer() -> None:
    """Probe the full tensorboard import chain to surface numpy/tensorflow incompatibilities early.

    When tensorboard is installed alongside a numpy-2.0-incompatible tensorflow, importing
    ``torch.utils.tensorboard`` raises ``AttributeError`` at module level (e.g. ``np.float_`` was
    removed in NumPy 2.0).  Calling this function inside the logger-construction try/except lets
    ``build_trainer`` degrade gracefully to CSV-only logging instead of crashing mid-training.

    Raises:
        ImportError: If the ``tensorboard`` package is absent.
        AttributeError: If ``torch.utils.tensorboard`` fails to import due to a NumPy 2.0 /
            tensorflow incompatibility.
    """
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401


# ---------------------------------------------------------------------------
# Notebook-safe spawn-based DDP
# ---------------------------------------------------------------------------
# ``ddp_notebook`` maps to fork-based DDP which is fundamentally unsafe:
# PyTorch's OpenMP thread pool (created during model construction) cannot
# survive fork() — the worker threads become zombie handles, causing
# "Invalid thread pool!" SIGABRT when the autograd engine initialises in
# the forked child.
#
# PTL considers ``start_method="spawn"`` incompatible with interactive
# environments and raises ``MisconfigurationException`` if used in Jupyter.
# However, PTL's own ``_wrapping_function`` is the entry-point for spawned
# children — no ``if __name__ == "__main__"`` guard is required — so spawn
# is perfectly safe here.
#
# Classes MUST live at module level (not inside a function) so that Python's
# pickle can serialise them for the spawned child processes.


if _MultiProcessingLauncher is not None:

    class _InteractiveSpawnLauncher(_MultiProcessingLauncher):
        """Spawn launcher that reports itself as interactive-compatible."""

        @property
        def is_interactive_compatible(self) -> bool:  # type: ignore[override]
            return True

else:
    _InteractiveSpawnLauncher = None


class _NotebookSpawnDDPStrategy(_DDPStrategy):
    """Spawn-based DDP strategy that works inside Jupyter / Kaggle notebooks."""

    def _configure_launcher(self) -> None:
        if self.cluster_environment is None:
            raise RuntimeError(
                "_NotebookSpawnDDPStrategy requires a cluster environment; "
                "ensure the strategy is initialised through PTL's Trainer."
            )
        if _InteractiveSpawnLauncher is None:
            raise RuntimeError(
                "Notebook spawn strategy requires "
                "pytorch_lightning.strategies.launchers.multiprocessing._MultiProcessingLauncher. "
                "Your installed PyTorch Lightning version changed this private API; "
                "pin/upgrade PTL to a compatible version in the supported >=2.6,<3 range."
            )
        self._launcher = _InteractiveSpawnLauncher(self, start_method=self._start_method)


def _is_distributed_strategy_requested(strategy: str) -> bool:
    """Return whether a TrainConfig strategy string requests distributed execution."""
    strategy_name = strategy.lower()
    return any(token in strategy_name for token in ("ddp", "fsdp", "deepspeed"))


def _accelerator_has_multiple_auto_devices(accelerator: str | None) -> bool:
    """Return whether PTL auto/all device resolution can select multiple devices."""
    accelerator_name = (accelerator or "auto").strip().lower()
    if accelerator_name in ("auto", "cuda", "gpu"):
        return torch.cuda.is_available() and torch.cuda.device_count() > 1
    return False


def _requests_multiple_devices(devices: int | str, accelerator: str | None = None) -> bool:
    """Return whether the configured devices value explicitly requests multiple devices."""
    if isinstance(devices, int):
        if devices == -1:
            return _accelerator_has_multiple_auto_devices(accelerator)
        return devices > 1
    devices_name = devices.strip().lower()
    if devices_name in ("auto", "-1"):
        return _accelerator_has_multiple_auto_devices(accelerator)
    if devices_name.isdigit():
        return int(devices_name) > 1
    if "," in devices_name:
        return len([entry for entry in devices_name.split(",") if entry.strip()]) > 1
    return False


def build_trainer(
    train_config: TrainConfig,
    model_config: ModelConfig,
    *,
    accelerator: str | None = None,
    **trainer_kwargs: Any,
) -> Trainer:
    """Assemble a PTL ``Trainer`` with the full RF-DETR callback and logger stack.

    Resolves training precision from ``model_config.amp`` and device capability, guards EMA against sharded strategies,
    wires conditional loggers, and applies promoted training knobs (sync_batchnorm, strategy).

    Args:
        train_config: Training hyperparameter configuration.
        model_config: Architecture configuration. Used for precision resolution
            (``model_config.amp``) and to guard against unsupported distributed
            configurations for keypoint models.
        accelerator: PTL accelerator string (e.g. ``"auto"``, ``"cpu"``, ``"gpu"``).
            Defaults to ``None`` which reads from ``train_config.accelerator`` (itself defaulting to ``"auto"``). Pass
            ``"cpu"`` to override auto-detection (e.g. when the caller explicitly requests CPU training via
            ``device="cpu"``).
        **trainer_kwargs: Extra keyword arguments forwarded to ``pytorch_lightning.Trainer``. Use this to pass
            PTL-native flags that are not exposed through ``TrainConfig``, for example::

                build_trainer(tc, mc, fast_dev_run=2)

            Most keys present in both ``trainer_kwargs`` and the built config dict are overridden by the value in
            ``trainer_kwargs``. Detection and segmentation models forward ``accumulate_grad_batches`` from
            ``train_config.grad_accum_steps`` and ``gradient_clip_val`` from ``train_config.clip_max_norm`` to the
            Trainer normally. Keypoint models force ``accumulate_grad_batches=1`` and ``gradient_clip_val=None``
            because ``RFDETRModelModule`` owns both operations under manual optimization; passing those keys for a
            keypoint config raises a ``UserWarning`` to make the override explicit.

    Returns:
        A configured ``pytorch_lightning.Trainer`` instance.
    """
    tc = train_config
    if accelerator is None:
        accelerator = tc.accelerator

    # --- Precision resolution ---
    def _resolve_precision() -> str:
        if not model_config.amp:
            if tc.amp_dtype != "auto":
                warnings.warn(
                    f"amp_dtype={tc.amp_dtype!r} has no effect when model_config.amp=False.",
                    UserWarning,
                    stacklevel=2,
                )
            return "32-true"
        # CPU accelerator: bf16 autocast on macOS CPU (Apple Silicon) is ~13x slower
        # than fp32 due to missing native bfloat16 kernels — no benefit, high cost.
        if accelerator == "cpu":
            return "32-true"
        # ``train_config.amp_dtype`` (a train() kwarg) lets callers pin the autocast dtype (see issue #1132):
        #   "auto" — bf16 on bf16-capable CUDA, fp16 otherwise (historical default);
        #   "fp16" — force "16-mixed" (e.g. deployment targets without bf16 support);
        #   "bf16" — force "bf16-mixed", falling back to fp16 with a warning when unsupported.
        # Unrecognised values are coerced to "auto" (with a warning) by TrainConfig validation.
        amp_dtype = tc.amp_dtype
        # Ampere+ GPUs support bf16-mixed which is scaler-free —
        # no GradScaler.scale/unscale/update overhead per optimizer step.
        # BF16 is safe for fine-tuning (pretrained weights loaded by default).
        # Training from random init with very small LR may underflow; pass
        # ``amp_dtype="fp16"`` if needed.
        #
        # Note: torch.cuda.is_available() and torch.cuda.is_bf16_supported() both
        # create a CUDA driver context in the parent process.  This is intentional
        # and safe for the multi-process launch modes we rely on here because we
        # avoid fork-based launching in notebook contexts (see
        # _NotebookSpawnDDPStrategy above), and spawn/subprocess-based launchers
        # start child processes with a fresh CUDA state regardless of what the
        # parent has initialised. If a fork-based path is ever added, this
        # precision check must be moved into the child process.
        if torch.cuda.is_available():
            if amp_dtype == "fp16":
                return "16-mixed"
            if amp_dtype == "bf16":
                if torch.cuda.is_bf16_supported():
                    return "bf16-mixed"
                _logger.warning(
                    "amp_dtype='bf16' was requested but this CUDA device does not support bfloat16; "
                    "falling back to fp16 ('16-mixed')."
                )
                warnings.warn(
                    "amp_dtype='bf16' was requested but this CUDA device does not support bfloat16; "
                    "falling back to fp16 ('16-mixed').",
                    UserWarning,
                    stacklevel=2,
                )
                return "16-mixed"
            # amp_dtype == "auto"
            return "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
        if torch.backends.mps.is_available():
            if amp_dtype == "bf16":
                _logger.warning(
                    "amp_dtype='bf16' is not applied on MPS; RF-DETR uses fp16 ('16-mixed') for MPS autocast."
                )
                warnings.warn(
                    "amp_dtype='bf16' is not applied on MPS; RF-DETR uses fp16 ('16-mixed') for MPS autocast.",
                    UserWarning,
                    stacklevel=2,
                )
            return "16-mixed"
        return "32-true"

    # --- Strategy + EMA sharding guard ---
    strategy = trainer_kwargs.get("strategy", tc.strategy)
    devices = trainer_kwargs.get("devices", tc.devices)
    num_nodes = trainer_kwargs.get("num_nodes", tc.num_nodes)
    strategy_name = strategy.strip().lower() if isinstance(strategy, str) else None
    has_keypoints = bool(model_config.use_grouppose_keypoints)
    if isinstance(tc, KeypointTrainConfig) != has_keypoints:
        raise ValueError(
            f"Config/model mismatch: isinstance(tc, KeypointTrainConfig)={isinstance(tc, KeypointTrainConfig)} "
            f"but model_config.use_grouppose_keypoints={model_config.use_grouppose_keypoints}. "
            "Pass KeypointTrainConfig for keypoint models and TrainConfig for detection models."
        )
    distributed_requested = (
        _is_distributed_strategy_requested(str(strategy))
        or num_nodes > 1
        or _requests_multiple_devices(devices, accelerator)
    )
    if has_keypoints and distributed_requested:
        # TODO(@keypoints-ddp): validate keypoint training under distributed strategies
        # before enabling keypoint distributed training.
        raise NotImplementedError(
            "Keypoint training currently does not support distributed execution "
            f"(strategy={strategy!r}, devices={devices!r}, num_nodes={num_nodes!r}). "
            "Use single-process training for now (for example strategy='auto', devices=1, num_nodes=1)."
        )

    # Transparently replace fork-based DDP with spawn-based DDP — see the
    # module-level comment block above _InteractiveSpawnLauncher for rationale.
    if strategy_name in ("ddp_notebook", "ddp_spawn"):
        strategy = _NotebookSpawnDDPStrategy(start_method="spawn", find_unused_parameters=True)
        _logger.info(
            "%s → spawn-based DDP to avoid OpenMP thread pool corruption after fork.",
            strategy_name,
        )
    elif strategy_name == "ddp" or (strategy_name == "auto" and distributed_requested):
        # DETR-family architectures can leave parameters unused on certain forward
        # steps under DDP, causing "It looks like your LightningModule has parameters
        # that were not used in producing the loss".  Sources include:
        #   - segmentation_head.sparse_forward() returning dict intermediates;
        #   - two-stage encoder query groups (group_detr ModuleLists) where per-group
        #     matcher assignment can leave groups without targets on low-annotation
        #     batches (issue #1093);
        #   - conditional auxiliary-loss branches.
        # Enabling find_unused_parameters lets DDP traverse the autograd graph after
        # each backward pass to identify which parameters contributed to the loss.
        # To opt out (e.g. configs with two_stage=False that never hit unused params),
        # pass strategy=DDPStrategy(find_unused_parameters=False) via trainer_kwargs.
        strategy = _DDPStrategy(find_unused_parameters=True)
        if strategy_name == "auto":
            _logger.info(
                "strategy='auto' with distributed execution → DDPStrategy(find_unused_parameters=True).",
            )
        else:
            _logger.info(
                "strategy='ddp' → DDPStrategy(find_unused_parameters=True).",
            )
    sharded = any(s in str(strategy).lower() for s in ("fsdp", "deepspeed"))
    enable_ema = bool(tc.use_ema) and not sharded
    if tc.use_ema and sharded:
        warnings.warn(
            f"EMA disabled: RFDETREMACallback is not compatible with sharded strategies "
            f"(strategy={strategy!r}). Set use_ema=False to suppress this warning.",
            UserWarning,
            stacklevel=2,
        )

    # --- Build callbacks ---
    callbacks = []

    if tc.progress_bar == "rich":
        callbacks.append(
            RichProgressBar(
                refresh_rate=5,
                theme=RichProgressBarTheme(metrics_format=".3e"),
            )
        )
    elif tc.progress_bar == "tqdm":
        callbacks.append(TQDMProgressBar(refresh_rate=5))

    if enable_ema:
        callbacks.append(
            RFDETREMACallback(
                decay=tc.ema_decay,
                tau=tc.ema_tau,
                update_interval_steps=tc.ema_update_interval,
            )
        )

    # Drop-path / dropout scheduling (vit_encoder_num_layers defaults to 12).
    if tc.drop_path > 0.0:
        callbacks.append(DropPathCallback(drop_path=tc.drop_path))

    # COCO mAP + F1 evaluation.
    callbacks.append(
        COCOEvalCallback(
            max_dets=tc.eval_max_dets,
            segmentation=model_config.segmentation_head,
            eval_interval=tc.eval_interval,
            log_per_class_metrics=tc.log_per_class_metrics,
            keypoint_oks_sigmas=tc.keypoint_oks_sigmas,
        )
    )

    # Latest resume checkpoint — overwritten every epoch.
    # Skip when checkpoint_interval == 1 to avoid duplicate ModelCheckpoint state_key.
    if tc.checkpoint_interval != 1:
        callbacks.append(
            ModelCheckpoint(
                dirpath=tc.output_dir,
                filename="last",
                every_n_epochs=1,
                save_top_k=1,
                enable_version_counter=False,
                auto_insert_metric_name=False,
                verbose=False,
            )
        )

    # Interval archive checkpoints — kept for the full run.
    callbacks.append(
        ModelCheckpoint(
            dirpath=tc.output_dir,
            filename="checkpoint_{epoch}",
            every_n_epochs=tc.checkpoint_interval,
            save_top_k=-1,
            enable_version_counter=False,
            auto_insert_metric_name=False,
            verbose=False,
        )
    )

    if has_keypoints:
        monitor_regular = "val/keypoint_map_50_95"
        early_stopping_monitor_ema = "val/ema_keypoint_map_50_95"
    elif model_config.segmentation_head:
        monitor_regular = "val/segm_mAP_50_95"
        early_stopping_monitor_ema = "val/ema_segm_mAP_50_95"
    else:
        monitor_regular = "val/mAP_50_95"
        early_stopping_monitor_ema = "val/ema_mAP_50_95"
    monitor_ema = early_stopping_monitor_ema if enable_ema else None

    best_model_smooth_alpha = tc.smooth_alpha

    # Best-model checkpointing — monitor EMA metric only when EMA is active and emitted.
    # PTL _reorder_callbacks moves all Checkpoint subclasses (including BestModelCallback)
    # to the end of the callback list; RFDETREarlyStopping (not a Checkpoint subclass) always
    # fires BEFORE BestModelCallback on every on_validation_end, regardless of append order.
    # The try/finally restore in BestModelCallback.on_validation_end guarantees EarlyStopping
    # always reads the raw (un-smoothed) metric value.
    callbacks.append(
        BestModelCallback(
            output_dir=tc.output_dir,
            monitor_regular=monitor_regular,
            monitor_ema=monitor_ema,
            run_test=tc.run_test,
            skip_best_epochs=tc.skip_best_epochs,
            smooth_alpha=best_model_smooth_alpha,
        )
    )

    # Optional early stopping.
    if tc.early_stopping:
        callbacks.append(
            RFDETREarlyStopping(
                patience=tc.early_stopping_patience,
                min_delta=tc.early_stopping_min_delta,
                use_ema=tc.early_stopping_use_ema,
                monitor_regular=monitor_regular,
                monitor_ema=early_stopping_monitor_ema,
                skip_best_epochs=tc.skip_best_epochs,
            )
        )

    # --- Build loggers ---
    # Each logger is guarded by a try/except because tensorboard, wandb, and mlflow
    # are optional dependencies (installed via the [metrics] extra).  A missing dep
    # emits a UserWarning instead of crashing.
    # CSVLogger is always enabled — no extra package required.
    # Produces metrics.csv in output_dir so there is always a log file.
    loggers: list = [CSVLogger(save_dir=tc.output_dir, name="", version="")]

    if tc.tensorboard:
        try:
            _try_import_tensorboard_summary_writer()
            loggers.append(
                TensorBoardLogger(
                    save_dir=tc.output_dir,
                    name="",
                    version="",
                )
            )
        except (ImportError, AttributeError) as exc:
            _logger.warning(
                "TensorBoard logging disabled: %s. "
                "If using NumPy 2.x, ensure your TensorBoard installation is NumPy 2.0 compatible "
                "(the failure can originate from tensorboard.compat.tensorflow_stub). "
                "Install TensorBoard with: pip install tensorboard",
                exc,
            )

    if tc.wandb:
        try:
            loggers.append(
                WandbLogger(
                    name=tc.run,
                    project=tc.project,
                    save_dir=tc.output_dir,
                )
            )
        except ModuleNotFoundError as exc:
            _logger.warning("WandB logging disabled: %s. Install with: pip install wandb", exc)

    if tc.mlflow:
        try:
            loggers.append(
                MLFlowLogger(
                    experiment_name=tc.project or "rfdetr",
                    run_name=tc.run,
                    save_dir=tc.output_dir,
                )
            )
        except ModuleNotFoundError as exc:
            _logger.warning("MLflow logging disabled: %s. Install with: pip install mlflow", exc)

    if tc.clearml:
        raise NotImplementedError("ClearML logging is not yet supported. Remove clearml=True from TrainConfig.")

    # --- Promoted config fields (T4-2 added these to TrainConfig) ---
    clip_max_norm: float = tc.clip_max_norm
    sync_bn: bool = tc.sync_bn

    # Manual optimization (currently scoped to keypoint models) owns gradient accumulation
    # and clipping inside ``RFDETRModelModule._step_optimizer`` so the box-count denominator
    # spans the full effective batch.  Detection and segmentation models keep Lightning's
    # automatic optimization, which means ``accumulate_grad_batches`` and ``gradient_clip_val``
    # must flow through to the Trainer as usual for them.
    manual_optimization = has_keypoints
    if manual_optimization:
        accumulate_grad_batches: int = 1
        gradient_clip_val: float | None = None
    else:
        accumulate_grad_batches = tc.grad_accum_steps
        gradient_clip_val = clip_max_norm

    trainer_config: dict[str, Any] = {
        "max_epochs": tc.epochs,
        "accelerator": accelerator,
        "devices": tc.devices,
        "num_nodes": tc.num_nodes,
        "strategy": strategy,
        "precision": _resolve_precision(),
        "accumulate_grad_batches": accumulate_grad_batches,
        "gradient_clip_val": gradient_clip_val,
        "sync_batchnorm": sync_bn,
        "callbacks": callbacks,
        "logger": loggers if loggers else False,
        "enable_progress_bar": tc.progress_bar is not None,
        "default_root_dir": tc.output_dir,
        "log_every_n_steps": 50,
        "deterministic": False,
    }
    trainer_config.update(trainer_kwargs)
    trainer_config["strategy"] = strategy
    if manual_optimization:
        # Re-apply manual-optimization invariants so a caller-supplied trainer_kwargs
        # value cannot silently re-enable Lightning-owned accumulation or clipping while
        # the module is doing its own.  Warn loudly so the override is visible — silent
        # coercion has historically masked subtle gradient-scaling bugs on this code path.
        for key in ("accumulate_grad_batches", "gradient_clip_val"):
            if key in trainer_kwargs:
                effective = "1" if key == "accumulate_grad_batches" else "None"
                alt = "grad_accum_steps" if key == "accumulate_grad_batches" else "clip_max_norm"
                warnings.warn(
                    f"build_trainer() ignored trainer_kwargs[{key!r}]={trainer_kwargs[key]!r} for a keypoint "
                    f"model. The model will train with {key}={effective} regardless of the value passed here "
                    f"because RFDETRModelModule owns gradient accumulation and clipping under manual "
                    f"optimization. To change the effective value, set TrainConfig.{alt} instead.",
                    UserWarning,
                    stacklevel=2,
                )
        trainer_config["accumulate_grad_batches"] = 1
        # gradient_clip_val=None here does NOT disable gradient clipping — clipping is
        # performed inside RFDETRModelModule._step_optimizer using train_config.clip_max_norm
        # (see src/rfdetr/training/module_model.py).  Under manual optimization the module
        # owns the clipping step; passing None to the PTL Trainer simply prevents PTL from
        # doing a second redundant clip on top of the module's own.
        trainer_config["gradient_clip_val"] = None
    return Trainer(**trainer_config)
