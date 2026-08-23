from __future__ import annotations

import gc
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .common import (
    LOGGER,
    MOSDataset,
    RunSpec,
    build_model,
    elapsed_text,
    gaussian_nll,
    make_loader,
    regression_metrics,
    save_json,
    set_seed,
    split_optimizer_parameters,
    torch_load,
)


def _autocast(enabled: bool):
    return torch.cuda.amp.autocast(enabled=enabled)


def _make_scaler(enabled: bool):
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _loss_for_experiment(
    experiment: str,
    mu: torch.Tensor,
    log_var: Optional[torch.Tensor],
    target: torch.Tensor,
) -> torch.Tensor:
    if experiment in {"mse", "dual_mse_check"}:
        return F.mse_loss(mu, target)
    if log_var is None:
        raise RuntimeError(f"{experiment} requires log variance")
    return gaussian_nll(mu, log_var, target)


def _gradient_audit(model, spec: RunSpec) -> Dict[str, Any]:
    head_linear = model.backbone.head_qs[1]
    head_grad = head_linear.weight.grad
    result: Dict[str, Any] = {
        "experiment": spec.experiment,
        "scope": spec.scope,
        "mean_head_row_grad_norm": None,
        "log_variance_head_row_grad_norm": None,
        "global_variance_grad_abs": None,
        "first_block_trainable": any(p.requires_grad for p in model.backbone.blocks[0].parameters()),
        "last_block_trainable": any(p.requires_grad for p in model.backbone.blocks[-1].parameters()),
    }
    if head_grad is None:
        raise AssertionError("Quality-head gradient is missing on the first batch")
    result["mean_head_row_grad_norm"] = float(head_grad[0].norm().detach().cpu())
    if result["mean_head_row_grad_norm"] <= 0:
        raise AssertionError("Mean-output row received zero gradient")
    if head_grad.shape[0] == 2:
        logvar_norm = float(head_grad[1].norm().detach().cpu())
        result["log_variance_head_row_grad_norm"] = logvar_norm
        if spec.experiment == "dual_mse_check" and logvar_norm > 1e-12:
            raise AssertionError("Dual-MSE negative control leaked gradient into log-variance row")
        if spec.experiment == "heteroscedastic_nll" and logvar_norm <= 0:
            raise AssertionError("Heteroscedastic log-variance row received zero gradient")
    if model.global_raw_log_var is not None:
        if model.global_raw_log_var.grad is None:
            raise AssertionError("Global log-variance parameter received no gradient")
        global_norm = float(model.global_raw_log_var.grad.abs().detach().cpu())
        result["global_variance_grad_abs"] = global_norm
        if global_norm <= 0:
            raise AssertionError("Global log-variance parameter received zero gradient")

    if spec.scope in {"head_only", "top1", "top2", "top4"} and result["first_block_trainable"]:
        raise AssertionError("Frozen lower Transformer block was unexpectedly trainable")
    if spec.scope != "head_only" and not result["last_block_trainable"]:
        raise AssertionError("Requested top/full fine-tuning did not unlock the last block")
    return result


def evaluate_model(
    model,
    loader,
    device: torch.device,
    experiment: str,
    amp_enabled: bool,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    records = []
    start = time.perf_counter()
    with torch.inference_mode():
        for images, mos, names, paths in loader:
            images = images.to(device, non_blocking=True)
            target = mos.to(device, non_blocking=True).view(-1)
            with _autocast(amp_enabled):
                mu, log_var = model(images)
            mu_np = mu.float().cpu().numpy()
            target_np = target.float().cpu().numpy()
            if log_var is None:
                log_var_np = np.full_like(mu_np, np.nan, dtype=np.float32)
            else:
                log_var_np = log_var.float().cpu().numpy()
            for index in range(len(mu_np)):
                lv = float(log_var_np[index])
                variance = float(math.exp(lv)) if math.isfinite(lv) else math.nan
                records.append(
                    {
                        "image_name": str(names[index]),
                        "image_path": str(paths[index]),
                        "mos": float(target_np[index]),
                        "mu": float(mu_np[index]),
                        "log_var": lv,
                        "variance": variance,
                        "std": math.sqrt(variance) if math.isfinite(variance) else math.nan,
                        "abs_error": abs(float(mu_np[index]) - float(target_np[index])),
                        "squared_error": (float(mu_np[index]) - float(target_np[index])) ** 2,
                    }
                )
    elapsed = time.perf_counter() - start
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError("Evaluation produced no predictions")
    metrics = regression_metrics(frame["mos"], frame["mu"])
    metrics["sample_count"] = int(len(frame))
    metrics["inference_seconds"] = float(elapsed)
    metrics["images_per_second"] = float(len(frame) / max(elapsed, 1e-9))
    if frame["log_var"].notna().all():
        metrics["gaussian_nll"] = float(
            np.mean(
                0.5
                * (
                    frame["log_var"].to_numpy()
                    + frame["squared_error"].to_numpy()
                    / frame["variance"].to_numpy()
                )
            )
        )
        metrics["mean_variance"] = float(frame["variance"].mean())
        metrics["mean_std"] = float(frame["std"].mean())
    return metrics, frame


def _scheduler_lambda(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_run(
    project_root: Path,
    run_dir: Path,
    config: Dict[str, Any],
    spec: RunSpec,
    frames: Dict[str, pd.DataFrame],
    smoke_test: bool = False,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    completion_file = run_dir / "completed.json"
    if completion_file.is_file():
        LOGGER.info("Skipping completed run: %s", spec.run_name)
        return torch_load_json(completion_file)

    train_cfg = config["training"]
    image_size = int(config["model"]["image_size"])
    epochs = 1 if smoke_test else int(train_cfg["epochs"])
    patience = epochs + 1 if smoke_test else int(train_cfg["early_stopping_patience"])
    batch_size = min(8, int(train_cfg["batch_size"])) if smoke_test else int(train_cfg["batch_size"])
    num_workers = 0 if smoke_test else int(train_cfg["num_workers"])

    set_seed(spec.seed, bool(train_cfg.get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    LOGGER.info("Starting %s on %s", spec.run_name, device)

    model, audit = build_model(project_root, config, spec.experiment, spec.scope)
    save_json(audit, run_dir / "parameter_audit.json")
    model = model.to(device)
    # Re-seed after model construction so single- and dual-output heads do not
    # shift augmentation, DropPath, or DataLoader random streams.
    set_seed(spec.seed, bool(train_cfg.get("deterministic", True)))

    run_frames = frames
    if smoke_test:
        run_frames = {
            split: frame.iloc[: min(len(frame), 16)].copy() for split, frame in frames.items()
        }
    datasets = {
        split: MOSDataset(frame, train=(split == "train"), image_size=image_size)
        for split, frame in run_frames.items()
    }
    loaders = {
        "train": make_loader(
            datasets["train"], batch_size, True, spec.seed, num_workers, device.type == "cuda"
        ),
        "val": make_loader(
            datasets["val"], batch_size, False, spec.seed, num_workers, device.type == "cuda"
        ),
        "test": make_loader(
            datasets["test"], batch_size, False, spec.seed, num_workers, device.type == "cuda"
        ),
    }

    backbone_params, head_params = split_optimizer_parameters(model)
    parameter_groups = []
    if backbone_params:
        parameter_groups.append(
            {"params": backbone_params, "lr": float(train_cfg["backbone_lr"]), "name": "backbone"}
        )
    if head_params:
        parameter_groups.append(
            {"params": head_params, "lr": float(train_cfg["head_lr"]), "name": "head"}
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: _scheduler_lambda(
            epoch, epochs, min(int(train_cfg["warmup_epochs"]), epochs)
        ),
    )
    scaler = _make_scaler(amp_enabled)

    best_srcc = -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history = []
    gradient_checked = False
    train_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(epochs):
        model.train()
        epoch_start = time.perf_counter()
        total_loss = 0.0
        total_examples = 0
        for images, mos, _, _ in loaders["train"]:
            images = images.to(device, non_blocking=True)
            target = mos.to(device, non_blocking=True).view(-1)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(amp_enabled):
                mu, log_var = model(images)
                loss = _loss_for_experiment(spec.experiment, mu, log_var, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss in {spec.run_name}, epoch {epoch + 1}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if not gradient_checked:
                gradient_report = _gradient_audit(model, spec)
                save_json(gradient_report, run_dir / "gradient_audit.json")
                gradient_checked = True
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(train_cfg["gradient_clip_norm"]),
            )
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(target)
            total_examples += len(target)

        scheduler.step()
        val_metrics, _ = evaluate_model(
            model, loaders["val"], device, spec.experiment, amp_enabled
        )
        epoch_seconds = time.perf_counter() - epoch_start
        record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(total_examples, 1),
            "val_srcc": val_metrics["srcc"],
            "val_plcc": val_metrics["plcc"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "backbone_lr": float(optimizer.param_groups[0]["lr"]),
            "head_lr": float(optimizer.param_groups[-1]["lr"]),
            "epoch_seconds": epoch_seconds,
        }
        if "gaussian_nll" in val_metrics:
            record["val_gaussian_nll"] = val_metrics["gaussian_nll"]
            record["val_mean_variance"] = val_metrics["mean_variance"]
        history.append(record)
        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
        LOGGER.info(
            "%s | epoch %02d/%02d | loss %.6f | val SRCC %.4f | PLCC %.4f | MAE %.4f",
            spec.run_name,
            epoch + 1,
            epochs,
            record["train_loss"],
            record["val_srcc"],
            record["val_plcc"],
            record["val_mae"],
        )

        if val_metrics["srcc"] > best_srcc + 1e-8:
            best_srcc = val_metrics["srcc"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "spec": asdict(spec),
                    "best_epoch": best_epoch,
                    "best_val_metrics": val_metrics,
                    "parameter_audit": audit,
                    "config": config,
                },
                run_dir / "best_model.pt",
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            LOGGER.info("Early stopping %s at epoch %d", spec.run_name, epoch + 1)
            break

    total_train_seconds = time.perf_counter() - train_start
    if best_epoch < 0:
        raise RuntimeError(f"No valid best checkpoint was produced for {spec.run_name}")
    checkpoint = torch_load(run_dir / "best_model.pt", device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    # The test split is touched exactly once, after model selection on validation.
    test_metrics, test_predictions = evaluate_model(
        model, loaders["test"], device, spec.experiment, amp_enabled
    )
    test_predictions.to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    val_metrics = checkpoint["best_val_metrics"]
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    )
    result: Dict[str, Any] = {
        "status": "completed",
        "run_name": spec.run_name,
        "experiment": spec.experiment,
        "scope": spec.scope,
        "seed": spec.seed,
        "role": spec.role,
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(history)),
        "train_seconds": float(total_train_seconds),
        "train_time": elapsed_text(total_train_seconds),
        "peak_gpu_memory_mb": peak_memory_mb,
        "total_parameters": audit["total_parameters"],
        "trainable_parameters": audit["trainable_parameters"],
        "trainable_ratio": audit["trainable_ratio"],
        "val": val_metrics,
        "test": test_metrics,
        "test_evaluations": 1,
        "smoke_test": bool(smoke_test),
    }
    save_json(result, completion_file)
    LOGGER.info(
        "Completed %s | test SRCC %.4f | PLCC %.4f | MAE %.4f | time %s",
        spec.run_name,
        test_metrics["srcc"],
        test_metrics["plcc"],
        test_metrics["mae"],
        result["train_time"],
    )

    del model, optimizer, scheduler, scaler, loaders, datasets
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def torch_load_json(path: Path) -> Dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
