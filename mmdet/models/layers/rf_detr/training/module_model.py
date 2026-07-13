# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""LightningModule for RF-DETR training and validation."""

from __future__ import annotations

import math
import random
import warnings
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 -- project-conventional alias (see AGENTS.md)
from pytorch_lightning import LightningModule, seed_everything

from mmdet.models.layers.rf_detr._namespace import _namespace_from_configs
from mmdet.models.layers.rf_detr.config import ModelConfig, TrainConfig
from mmdet.models.layers.rf_detr.datasets.coco import compute_multi_scale_scales
from mmdet.models.layers.rf_detr.models.lwdetr import build_criterion_from_config, build_model_from_config
from mmdet.models.layers.rf_detr.models.weights import apply_lora, interpolate_position_embeddings, load_pretrain_weights
from mmdet.models.layers.rf_detr.training.param_groups import get_param_dict
from mmdet.models.layers.rf_detr.utilities.logger import get_logger

logger = get_logger()

_TRAIN_PROGRESS_LOSS_ALIASES: dict[str, str] = {
    "loss_ce": "loss_cls",
    "loss_bbox": "loss_box",
    "loss_giou": "loss_giou",
    "loss_mask_ce": "mask_ce",
    "loss_mask_dice": "mask_dice",
    "loss_keypoints_l1": "kp_l1",
    "loss_keypoints_findable": "kp_find",
    "loss_keypoints_visible": "kp_vis",
    "loss_keypoints_nll": "kp_nll",
}


class RFDETRModelModule(LightningModule):
    """LightningModule wrapping the RF-DETR model and training loop.

    Args:
        model_config: Architecture configuration.
        train_config: Training hyperparameter configuration.
    """

    def __init__(self, model_config: ModelConfig, train_config: TrainConfig) -> None:
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        # Manual optimization is enabled only for keypoint models so that the box-count
        # normalizer can be accumulated across grad-accum microbatches. Detection and
        # segmentation use Lightning's automatic optimization (PTL handles accumulation,
        # AMP, and gradient clipping), which keeps their step semantics unchanged from
        # the pre-fix/scaling behaviour.
        self._use_manual_optimization: bool = bool(getattr(model_config, "use_grouppose_keypoints", False))
        self.automatic_optimization = not self._use_manual_optimization
        self._accumulated_box_normalizer: torch.Tensor | None = None
        # Allow partial state-dict loading when resuming from a .pth checkpoint
        # (which contains only model weights, not criterion/postprocess state).
        self.strict_loading = False

        # Model, criterion, and postprocessor.
        self.model = build_model_from_config(model_config, train_config)
        if model_config.pretrain_weights is not None:
            # Canonical loader handles PE interpolation, PTL .ckpt normalisation,
            # per-group query slicing, class-name extraction, partial-load warnings,
            # and writes any auto-aligned ``num_classes`` back onto ``model_config``.
            load_pretrain_weights(self.model, self.model_config)
            if model_config.use_grouppose_keypoints:
                # Older model shims may omit the keypoint reset hook; call it only when implemented.
                reset_keypoint_gaussian_parameters = getattr(self.model, "reset_keypoint_gaussian_parameters", None)
                if callable(reset_keypoint_gaussian_parameters):
                    reset_keypoint_gaussian_parameters()
                    logger.info(
                        "Reset keypoint Gaussian precision outputs to unit values after pretrained weight load."
                    )
        if model_config.backbone_lora:
            apply_lora(self.model)

        # Build criterion/postprocessors after potential num_classes alignment so
        # they are constructed with a config that matches the current model head.
        self.criterion, self.postprocess = build_criterion_from_config(self.model_config, self.train_config)

        # torch.compile is opt-in: set model_config.compile=True to enable.
        # Only enabled on CUDA; MPS and CPU do not benefit from compilation.
        # Use the fork-safe DEVICE constant instead of torch.cuda.is_available(),
        # which creates a CUDA driver context that breaks fork-based DDP.
        from mmdet.models.layers.rf_detr.config import DEVICE

        accelerator = str(train_config.accelerator).lower()
        uses_cuda_accelerator = accelerator in {"auto", "gpu", "cuda"}
        compile_enabled = (
            model_config.compile and DEVICE == "cuda" and uses_cuda_accelerator and not train_config.multi_scale
        )
        if model_config.compile and train_config.multi_scale:
            logger.info("Disabling torch.compile because multi_scale=True introduces dynamic input shapes.")
        if compile_enabled:
            # dynamic=True: one compiled graph handles all multi-scale input sizes instead
            # of recompiling per (H, W) pair. suppress_errors=True: if inductor can't
            # compile a subgraph (e.g. bicubic backward with symbolic shapes), it falls
            # back to eager mode for that subgraph rather than crashing.
            # capture_scalar_outputs=True: include Tensor.item() calls
            # (gen_encoder_output_proposals / ms_deform_attn use spatial-shape .item()
            # as Python slice indices). Safe with dynamic=True because item() results
            # are backed symbols derived from input shapes — not unbacked symbols that
            # would cause PendingUnbackedSymbolNotFound (which only occurs without dynamic).
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.capture_scalar_outputs = True
            self.model = torch.compile(self.model, dynamic=True)

    # ------------------------------------------------------------------
    # PTL lifecycle hooks
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        """Seed RNGs at fit start when ``TrainConfig.seed`` is set.

        This avoids hidden global side-effects in ``build_trainer`` while still preserving deterministic training
        behaviour for actual fit runs.
        """
        if self.train_config.seed is not None:
            seed_everything(self.train_config.seed + self.global_rank, workers=True)

    def on_train_start(self) -> None:
        """Normalize restored fused-optimizer state before the first training step.

        Lightning restores optimizer state after ``on_fit_start``.  Fused AdamW is strict about the dtype, device, and
        layout of its moment tensors, so resuming from a checkpoint can fail if Lightning rehydrates those tensors in a
        layout that no longer matches the live parameters.  Recasting same-shaped floating-point tensors here keeps the
        resumed optimizer compatible without discarding the saved momentum state.
        """
        if not self._use_fused_optimizer:
            return

        try:
            optimizers = self.optimizers(use_pl_optimizer=False)
        except RuntimeError:
            return
        if optimizers is None:
            return
        if isinstance(optimizers, list):
            optimizer_list = optimizers
        else:
            optimizer_list = [optimizers]

        normalized_tensors = 0
        for optimizer in optimizer_list:
            if not isinstance(optimizer, torch.optim.Optimizer):
                optimizer = getattr(optimizer, "optimizer", optimizer)
            if isinstance(optimizer, torch.optim.Optimizer):
                normalized_tensors += self._normalize_optimizer_state(optimizer)
        if normalized_tensors and getattr(self.trainer, "is_global_zero", True):
            logger.info(
                "Normalized %d restored fused AdamW state tensors after checkpoint resume.",
                normalized_tensors,
            )

    def on_train_batch_start(self, batch: tuple, batch_idx: int) -> None:
        """Apply optional multi-scale resize to the incoming batch.

        Modifications to ``batch`` (in-place on ``NestedTensor``) are visible in ``training_step`` because they share
        the same object.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Index of the current batch within the epoch.
        """
        tc = self.train_config
        mc = self.model_config

        if tc.multi_scale and not tc.do_random_resize_via_padding:
            samples, _ = batch
            scales = compute_multi_scale_scales(mc.resolution, tc.expanded_scales, mc.patch_size, mc.num_windows)
            step = self.trainer.global_step
            # Use a step-local generator so the scale choice is deterministic and DDP-consistent
            # without reseeding the process-global RNG on every batch.
            scale = random.Random(step).choice(scales)
            with torch.no_grad():
                samples.tensors = F.interpolate(samples.tensors, size=scale, mode="bilinear", align_corners=False)
                samples.mask = (
                    F.interpolate(samples.mask.unsqueeze(1).float(), size=scale, mode="nearest").squeeze(1).bool()
                )

    def on_train_epoch_start(self) -> None:
        """Reset the accumulated box normalizer at the start of every training epoch.

        Lightning may reuse the module across epochs without calling ``_step_optimizer`` at the boundary (for example
        when an epoch ends mid-accumulation window with a non-divisible batch count). Clearing the accumulator here
        guarantees the manual-optimization path always starts each epoch from a known state, so the first microbatch's
        gradients are scaled by its own box count and not by a stale previous-epoch denominator.

        This is a no-op for non-keypoint models because they use Lightning's automatic optimization path and never
        populate ``self._accumulated_box_normalizer``.

        Note: on finite datasets the final-batch fallback in ``_should_step_optimizer`` always flushes a partial
        trailing window, so this reset is the only change needed.  On IterableDatasets (infinite
        ``num_training_batches``) a partial window may survive epoch end with un-stepped gradients; those are
        discarded here and the optimizer is zeroed so the first microbatch of the new epoch starts from a clean state.
        """
        if self._accumulated_box_normalizer is not None:
            # Discard any partial accumulation window that survived the epoch boundary
            # (only possible for IterableDatasets where num_training_batches is infinite).
            try:
                opts = self.optimizers()
                for opt in opts if isinstance(opts, list) else [opts]:
                    opt.zero_grad()
            except RuntimeError:
                pass  # Not attached to Trainer (unit-test context); nothing to zero.
        self._accumulated_box_normalizer = None

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor | dict[str, Any]:
        """Compute loss for one training step and log metrics.

        PTL handles AMP (``precision``) without a manual ``GradScaler``. Keypoint models perform manual optimization so
        box-count loss normalization is based on the full accumulated effective batch rather than each microbatch
        independently; detection and segmentation models keep Lightning's automatic optimization path.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Batch index within the epoch.

        Returns:
            Scalar loss tensor by default. When ``compute_train_metrics=True``,
            returns a Lightning-compatible dict containing ``loss`` plus
            detached postprocessed predictions for train mAP logging.
        """
        samples, targets = batch
        batch_size = len(targets)
        outputs = self.model(samples, targets)
        if self._use_manual_optimization:
            loss_dict, raw_loss, normalizer = self._compute_train_losses(outputs, targets)
            loss_for_backward = self._scale_loss_for_accumulation(raw_loss, normalizer)
        else:
            loss_dict = self.criterion(outputs, targets)
            loss_for_backward = None
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
        # Automatic optimization path: divide by accumulate_grad_batches so the accumulated
        # gradient matches a single large batch, matching the legacy engine.  PTL accumulates
        # full-scale gradients by default; dividing here keeps the effective LR identical.
        accumulate_grad_batches = max(1, int(self.trainer.accumulate_grad_batches))
        loss_for_return = loss if self._use_manual_optimization else loss / accumulate_grad_batches
        train_log_sync_dist = bool(self.train_config.train_log_sync_dist)
        train_log_on_step = bool(self.train_config.train_log_on_step)
        self.log_dict(
            {f"train/{k}": v for k, v in loss_dict.items()},
            on_step=train_log_on_step,
            on_epoch=True,
            sync_dist=train_log_sync_dist,
            batch_size=batch_size,
        )
        self.log(
            "train/loss",
            loss,
            prog_bar=False,
            on_step=train_log_on_step,
            on_epoch=True,
            sync_dist=train_log_sync_dist,
            batch_size=batch_size,
        )
        self._log_train_progress_metrics(loss, loss_dict, batch_size=batch_size)
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        # Optimizer may have multiple param groups with different LRs (e.g., backbone/decoder).
        # Preserve the first group's LR for backward compatibility, but also log the
        # min/max across all groups so the progress bar reflects the full schedule.
        group_lrs = [pg["lr"] for pg in optimizer.param_groups if "lr" in pg]
        if group_lrs:
            base_lr = group_lrs[0]
            min_lr = min(group_lrs)
            max_lr = max(group_lrs)
            self.log("train/lr", base_lr, prog_bar=False, on_step=True, on_epoch=False)
            self.log("train/lr_min", min_lr, prog_bar=False, on_step=True, on_epoch=False)
            self.log("train/lr_max", max_lr, prog_bar=False, on_step=True, on_epoch=False)
        if self._use_manual_optimization:
            self.manual_backward(loss_for_backward)
            if self._should_step_optimizer(batch_idx):
                self._step_optimizer(optimizer)
        if self.train_config.compute_train_metrics:
            with torch.no_grad():
                orig_sizes = torch.stack([t["orig_size"] for t in targets])
                # Slice to group-0 queries only — mirrors the eval-mode path in
                # lwdetr.py that trims refpoint_embed to [:num_queries]. Without
                # this, training mode emits group_detr×num_queries queries (e.g.
                # 13×300=3900) and postprocess top-k selection draws from all
                # groups, producing OKS/mAP values ~50× below true accuracy.
                nq = self.model_config.num_queries
                # Only include tensor-valued keys — pred_masks is a dict in
                # train mode (sparse_forward) and postprocess cannot handle it.
                inference_outputs = {
                    k: v[:, :nq] if v.ndim >= 2 else v
                    for k, v in outputs.items()
                    if k in ("pred_logits", "pred_boxes", "pred_masks", "pred_keypoints")
                    and isinstance(v, torch.Tensor)
                }
                results = self.postprocess(inference_outputs, orig_sizes)
            return {
                "loss": loss_for_return.detach() if self._use_manual_optimization else loss_for_return,
                "results": self._detach_results(results),
                "targets": targets,
            }
        return loss_for_return.detach() if self._use_manual_optimization else loss_for_return

    def _compute_train_losses(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Compute normalized losses for logging and raw weighted loss for backward.

        Args:
            outputs: Model output dictionary.
            targets: Target dictionaries for the current batch.

        Returns:
            A tuple of normalized loss dictionary, unnormalized weighted loss numerator, and box normalizer.
        """
        weight_dict = self.criterion.weight_dict
        if not getattr(self.criterion, "supports_loss_normalizer_override", False):
            raise ValueError(
                f"{type(self.criterion).__name__}.supports_loss_normalizer_override is False; "
                "manual optimization (keypoint models) requires a criterion that accepts a "
                "num_boxes keyword argument. Set supports_loss_normalizer_override = True on "
                "your criterion subclass and implement the num_boxes parameter in forward()."
            )
        normalizer = self.criterion.num_boxes_for_targets(outputs, targets)
        numerator_loss_dict = self.criterion(outputs, targets, num_boxes=torch.ones_like(normalizer))
        # Keys in weight_dict are loss terms whose criterion implementation divides by num_boxes
        # (so passing num_boxes=1.0 yields raw numerators that we divide by normalizer here).
        # Keys outside weight_dict (e.g. "class_error", "cardinality_error") are diagnostics
        # that do NOT divide by num_boxes internally — they are passed through unchanged.
        # If a future loss term divides by num_boxes AND is omitted from weight_dict, its
        # logged value will be on a different scale than the keypoint path; verify when adding
        # new criterion terms.
        loss_dict = {
            key: value / normalizer if key in weight_dict else value for key, value in numerator_loss_dict.items()
        }
        raw_loss = sum(numerator_loss_dict[k] * weight_dict[k] for k in numerator_loss_dict if k in weight_dict)
        return loss_dict, raw_loss, normalizer

    def _scale_loss_for_accumulation(
        self,
        raw_loss: torch.Tensor,
        normalizer: torch.Tensor,
    ) -> torch.Tensor:
        """Scale the current numerator loss by the accumulated box denominator.

        Args:
            raw_loss: Current microbatch weighted loss numerator.
            normalizer: Current microbatch box denominator.

        Returns:
            Loss scalar to pass to ``manual_backward``.
        """
        normalizer = normalizer.detach()
        previous_normalizer = self._accumulated_box_normalizer
        accumulated_normalizer = normalizer if previous_normalizer is None else previous_normalizer + normalizer
        if previous_normalizer is not None:
            self._rescale_accumulated_gradients(previous_normalizer / accumulated_normalizer)
        self._accumulated_box_normalizer = accumulated_normalizer.detach()
        return raw_loss / accumulated_normalizer

    def _rescale_accumulated_gradients(self, scale: torch.Tensor) -> None:
        """Rescale gradients already accumulated in the current optimizer window.

        Args:
            scale: Multiplicative factor that converts previous gradients from the old denominator to the new one.
        """
        for parameter in self.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale.to(device=parameter.grad.device, dtype=parameter.grad.dtype))

    def _should_step_optimizer(self, batch_idx: int) -> bool:
        """Return whether the current batch closes an optimizer accumulation window.

        The optimizer steps when either:

        - The current batch closes a complete ``grad_accum_steps`` window
          (``(batch_idx + 1) % grad_accum_steps == 0``), or
        - This is the final batch of the epoch and a partial accumulation window
          is still open, so the trailing microbatches are not silently dropped.

        Lightning's ``Trainer.num_training_batches`` may be reported as ``float('inf')``
        for iterable / streaming datasets where the epoch length is unknown. In that case
        only the modulo path can ever close the window — the final-batch fallback is
        skipped because ``batch_idx + 1`` can never reach infinity.

        Args:
            batch_idx: Batch index within the epoch.

        Returns:
            ``True`` when the optimizer should step after this batch.
        """
        accum_steps = max(1, int(self.train_config.grad_accum_steps))
        if (batch_idx + 1) % accum_steps == 0:
            return True
        num_training_batches = getattr(self.trainer, "num_training_batches", None)
        return (
            isinstance(num_training_batches, (int, float))
            and math.isfinite(num_training_batches)
            and batch_idx + 1 >= num_training_batches
        )

    def _step_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Clip gradients, step optimizer and scheduler, then reset accumulation state.

        Args:
            optimizer: Optimizer returned by Lightning.
        """
        trainer_gradient_clip_val = getattr(self.trainer, "gradient_clip_val", None)
        if trainer_gradient_clip_val is None:
            gradient_clip_val = self.train_config.clip_max_norm
        elif isinstance(trainer_gradient_clip_val, (int, float)):
            gradient_clip_val = trainer_gradient_clip_val
        else:
            gradient_clip_val = None
        gradient_clip_algorithm = getattr(self.trainer, "gradient_clip_algorithm", None)
        if not isinstance(gradient_clip_algorithm, str):
            gradient_clip_algorithm = None
        if gradient_clip_val is not None and gradient_clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm,
            )
        optimizer.step()
        optimizer.zero_grad()
        self._step_lr_scheduler()
        self._accumulated_box_normalizer = None

    def _step_lr_scheduler(self) -> None:
        """Step Lightning's scheduler object when one is configured."""
        try:
            scheduler = self.lr_schedulers()
        except (AttributeError, RuntimeError):
            return
        if scheduler is None:
            return
        schedulers = scheduler if isinstance(scheduler, list) else [scheduler]
        for scheduler_item in schedulers:
            scheduler_item.step()

    @staticmethod
    def _detach_results(results: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        """Detach postprocessed result tensors before handing them to callbacks.

        Args:
            results: Per-image postprocessed prediction dictionaries.

        Returns:
            Per-image dictionaries with tensor values detached from the graph.
        """
        return [
            {key: value.detach() if torch.is_tensor(value) else value for key, value in result.items()}
            for result in results
        ]

    def _log_train_progress_metrics(
        self,
        loss: torch.Tensor,
        loss_dict: dict[str, torch.Tensor],
        *,
        batch_size: int,
    ) -> None:
        """Log compact per-step convergence metrics for the progress bar only.

        Args:
            loss: Unscaled aggregate training loss.
            loss_dict: Raw criterion loss dictionary.
            batch_size: Current batch size used by Lightning for metric reduction metadata.
        """
        self.log(
            "loss",
            loss,
            prog_bar=True,
            logger=False,
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )
        for loss_name, progress_name in _TRAIN_PROGRESS_LOSS_ALIASES.items():
            value = loss_dict.get(loss_name)
            if value is None:
                continue
            self.log(
                progress_name,
                value,
                prog_bar=True,
                logger=False,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )

    def _log_val_loss_metrics(
        self,
        loss: torch.Tensor,
        loss_dict: dict[str, torch.Tensor],
        *,
        batch_size: int,
    ) -> None:
        """Log aggregate and component validation losses.

        Args:
            loss: Aggregate weighted validation loss.
            loss_dict: Raw criterion loss dictionary.
            batch_size: Current batch size used by Lightning for metric reduction metadata.
        """
        self.log_dict(
            {f"val/{k}": v for k, v in loss_dict.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=batch_size)

    def validation_step(self, batch: tuple, batch_idx: int) -> dict[str, Any]:
        """Run forward pass and postprocess for one validation step.

        Returns raw results and targets so ``COCOEvalCallback`` can accumulate them across the epoch via
        ``on_validation_batch_end``.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Batch index within the validation epoch.

        Returns:
            Dict with ``results`` (postprocessed predictions) and ``targets``.
        """
        samples, targets = batch
        outputs = self.model(samples)
        if self.train_config.compute_val_loss:
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
            self._log_val_loss_metrics(loss, loss_dict, batch_size=len(targets))

        orig_sizes = torch.stack([t["orig_size"] for t in targets])
        results = self.postprocess(outputs, orig_sizes)
        return {"results": results, "targets": targets}

    @property
    def _use_fused_optimizer(self) -> bool:
        """Return whether fused AdamW should be used for the current training configuration.

        Fused AdamW is only safe when the trainer's actual precision is a BF16 variant.  Checking GPU capability alone
        (``is_bf16_supported()``) is
        insufficient: on Ampere+ hardware that flag is always ``True`` even when
        the trainer is configured for ``32-true``, which causes a ``params, grads, exp_avgs, and exp_avg_sqs must have
        same dtype, device, and layout`` crash in DDP because gradient bucket views have non-matching strides in FP32.

        Returns:
            ``True`` when fused AdamW is both requested and safe to use.

        Examples:
            >>> from unittest.mock import patch
            >>> module = RFDETRModelModule.__new__(RFDETRModelModule)
            >>> module.model_config = type("Cfg", (), {"fused_optimizer": True})()
            >>> with patch("torch.cuda.is_available", return_value=False):
            ...     module._use_fused_optimizer
            False
        """
        return (
            self.model_config.fused_optimizer
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            and str(self.trainer.precision) in {"bf16-mixed", "bf16", "bf16-true"}
        )

    def configure_optimizers(self) -> dict[str, Any]:
        """Build AdamW optimizer with layer-wise LR decay and LambdaLR scheduler.

        Uses ``trainer.estimated_stepping_batches`` for total step count so cosine annealing covers the full training
        run regardless of dataset size or accumulation settings.

        Returns:
            PTL optimizer config dict with optimizer and step-interval scheduler.
        """
        tc = self.train_config
        ns = _namespace_from_configs(self.model_config, tc)

        # Unwrap torch.compile's OptimizedModule so get_param_dict sees the
        # original module's named_parameters() — compiled wrapper can cause
        # name-prefix mismatches that put the same tensor in multiple groups.
        model_for_params = getattr(self.model, "_orig_mod", self.model)
        param_dicts = get_param_dict(ns, model_for_params)
        param_dicts = [p for p in param_dicts if p["params"].requires_grad]
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=tc.lr,
            weight_decay=tc.weight_decay,
            fused=self._use_fused_optimizer,
        )

        # ``trainer.estimated_stepping_batches`` is reported in *microbatch* units when
        # the keypoint path runs with ``Trainer(accumulate_grad_batches=1)`` and manages
        # accumulation manually. ``LambdaLR.step()`` is called once per optimizer-step
        # (i.e. every ``grad_accum_steps`` microbatches), so the schedule must be sized
        # in optimizer-step units rather than microbatches; otherwise warmup and cosine
        # decay finish ``grad_accum_steps``× too early. Detection / segmentation models
        # still rely on Lightning's automatic optimization, where PTL already accounts
        # for ``accumulate_grad_batches`` inside ``estimated_stepping_batches`` and the
        # division below is a no-op (``grad_accum_steps`` would be 1 in that path).
        grad_accum_steps = max(1, int(tc.grad_accum_steps))
        microbatches = int(self.trainer.estimated_stepping_batches)
        # _should_step_optimizer steps the final partial window at epoch end, so the true
        # number of optimizer steps is ceil(microbatches / grad_accum_steps).  Using floor
        # would undercount when the epoch is not evenly divisible, causing warmup / cosine
        # schedules to finish one step earlier than the last actual step fires.
        total_steps = (
            max(1, math.ceil(microbatches / grad_accum_steps)) if self._use_manual_optimization else microbatches
        )
        steps_per_epoch = max(1, total_steps // tc.epochs)
        warmup_steps = int(steps_per_epoch * tc.warmup_epochs)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if tc.lr_scheduler == "cosine":
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return tc.lr_min_factor + (1 - tc.lr_min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
            # Step decay: drop by 10× after lr_drop epochs.
            if current_step < tc.lr_drop * steps_per_epoch:
                return 1.0
            return 0.1

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def clip_gradients(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Override PTL gradient clipping to support fused AdamW.

        PTL's AMP precision plugin refuses to clip gradients when the optimizer declares it handles unscaling internally
        (fused=True).  When fused is active we are on BF16 (no GradScaler) so ``clip_grad_norm_`` is correct.  For the
        non-fused path (FP16 + GradScaler or FP32) we delegate to ``super()`` to preserve scaler-aware unscaling.

        Args:
            optimizer: The current optimizer.
            gradient_clip_val: Maximum gradient norm.
            gradient_clip_algorithm: Clipping algorithm; forwarded to super()
                for the non-fused path.
        """
        if self._use_fused_optimizer:
            if gradient_clip_val and gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_val)
        else:
            super().clip_gradients(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm,
            )

    @staticmethod
    def _normalize_optimizer_state(optimizer: torch.optim.Optimizer) -> int:
        """Cast restored floating-point optimizer state tensors to match live parameters.

        Args:
            optimizer: AdamW optimizer whose state may have been rehydrated with a mismatched dtype or layout.

        Returns:
            Number of state tensors that were reallocated to match the current parameter layout.
        """
        normalized = 0
        for group in optimizer.param_groups:
            for param in group["params"]:
                state = optimizer.state.get(param)
                if not state:
                    continue
                for key, value in list(state.items()):
                    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                        continue
                    if value.shape != param.shape:
                        continue
                    if value.device == param.device and value.dtype == param.dtype and value.stride() == param.stride():
                        continue
                    restored = torch.empty_like(param)
                    restored.copy_(value.to(device=param.device, dtype=param.dtype))
                    state[key] = restored
                    normalized += 1
        return normalized

    def test_step(self, batch: tuple, batch_idx: int) -> dict[str, Any]:
        """Run forward pass and postprocess for one test step.

        Mirrors :meth:`validation_step` so ``COCOEvalCallback`` can accumulate results via ``on_test_batch_end`` when
        ``trainer.test()`` is called (e.g. from :class:`~rfdetr.training.callbacks.BestModelCallback` at end of
        training).

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Batch index within the test epoch.

        Returns:
            Dict with ``results`` (postprocessed predictions) and ``targets``.
        """
        samples, targets = batch
        outputs = self.model(samples)
        if self.train_config.compute_test_loss:
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
            self.log("test/loss", loss, sync_dist=True, batch_size=len(targets))

        orig_sizes = torch.stack([t["orig_size"] for t in targets])
        results = self.postprocess(outputs, orig_sizes)
        return {"results": results, "targets": targets}

    def predict_step(self, batch: tuple, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Run inference on a preprocessed batch and return postprocessed results.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Batch index.
            dataloader_idx: Index of the predict dataloader.

        Returns:
            Postprocessed detection results from ``PostProcess``.
        """
        samples, targets = batch
        with torch.no_grad():
            outputs = self.model(samples)
        orig_sizes = torch.stack([t["orig_size"] for t in targets])
        return self.postprocess(outputs, orig_sizes)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Auto-detect legacy formats and reconcile PE shapes at checkpoint load time.

        PTL calls this hook before applying ``checkpoint["state_dict"]`` to the module.  Three normalisation steps are
        applied in order:

        1. **Raw legacy format** — a ``*.pth`` file loaded directly by
           ``Trainer`` (e.g. via ``ckpt_path=``).  Recognised by the presence of ``"model"`` without ``"state_dict"``.
           The state dict is rewritten in-place with the ``"model."`` prefix so PTL can apply it normally.

        2. **Positional-embedding interpolation** — when the checkpoint was
           saved at a different image resolution than the current model, the DINOv2 ``position_embeddings`` tensor shape
           will mismatch. :func:`~rfdetr.models.weights.interpolate_position_embeddings` is called to bicubic-resize the
           PE to ``model_config.positional_encoding_size`` before PTL applies the state dict.  Regression fix for
           :issue:`998`.

        3. **Converted format** — a file produced by
           :func:`~rfdetr.training.checkpoint.convert_legacy_checkpoint` that already has ``"state_dict"`` but also
           carries ``"legacy_ema_state_dict"``.  The EMA weights are stashed on ``self._pending_legacy_ema_state`` for
           optional restoration by :class:`~rfdetr.training.callbacks.ema.RFDETREMACallback`.

        Note:
            This hook only fires on ``Trainer(ckpt_path=...)`` resume paths. Fresh-train bootstrap from a
            ``pretrain_weights`` checkpoint runs through :func:`~rfdetr.models.weights.load_pretrain_weights` during
            ``__init__`` instead — that helper performs its own PTL ``.ckpt`` normalisation (``state_dict`` → ``model``
            key, ``_orig_mod`` strip) and PE interpolation, so the two code paths intentionally do not share state.

        Args:
            checkpoint: Checkpoint dict passed in by PTL (mutated in-place).
        """
        # Raw legacy .pth: no "state_dict" key — build it from "model".
        if "model" in checkpoint and "state_dict" not in checkpoint:
            checkpoint["state_dict"] = {"model." + k: v for k, v in checkpoint["model"].items()}

        # Interpolate DINOv2 positional embeddings when the checkpoint was saved
        # at a different resolution than the current model.  PTL applies
        # checkpoint["state_dict"] immediately after this hook, so the shapes
        # must already match at this point.  Regression: #998.
        if "state_dict" in checkpoint:
            interpolate_position_embeddings(
                checkpoint["state_dict"],
                self.model_config.positional_encoding_size,
            )

        # Stash legacy EMA weights for RFDETREMACallback.setup(), which restores
        # them into AveragedModel when resuming from converted legacy checkpoints.
        if "legacy_ema_state_dict" in checkpoint:
            self._pending_legacy_ema_state = checkpoint["legacy_ema_state_dict"]
            warnings.warn(
                "Checkpoint contains legacy EMA weights (`legacy_ema_state_dict`). "
                "Add RFDETREMACallback to your trainer callbacks to restore them; "
                "without it the stashed weights will be ignored.",
                UserWarning,
                stacklevel=2,
            )

    def reinitialize_detection_head(self, num_classes: int) -> None:
        """Reinitialize the detection head for a new class count.

        Args:
            num_classes: New number of classes (excluding background).
        """
        self.model.reinitialize_detection_head(num_classes)
