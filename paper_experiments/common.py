from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


LOGGER = logging.getLogger("fiqa_paper")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_split(split_dir: str) -> Tuple[Path, Path]:
    base = Path(split_dir)
    return base / "labels.csv", base / "img"


def read_labels(csv_path: Path, image_dir: Path, split_name: str) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"{split_name}: labels.csv does not exist: {csv_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"{split_name}: img directory does not exist: {image_dir}")

    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    if raw.shape[1] < 2:
        raise ValueError(f"{split_name}: labels.csv must contain at least two columns")
    frame = raw.iloc[:, :2].copy()
    frame.columns = ["image_name", "mos"]
    frame["image_name"] = frame["image_name"].astype(str).str.strip()
    frame["mos"] = pd.to_numeric(frame["mos"], errors="raise").astype(float)
    if frame.empty:
        raise ValueError(f"{split_name}: labels.csv is empty")
    if frame["image_name"].eq("").any():
        raise ValueError(f"{split_name}: empty image filename found")
    if frame["mos"].isna().any() or not np.isfinite(frame["mos"].to_numpy()).all():
        raise ValueError(f"{split_name}: MOS contains NaN or infinite values")

    duplicate_count = int(frame["image_name"].duplicated().sum())
    if duplicate_count:
        examples = frame.loc[frame["image_name"].duplicated(), "image_name"].head(5).tolist()
        raise ValueError(
            f"{split_name}: {duplicate_count} duplicate filenames found; examples={examples}"
        )

    frame["image_path"] = frame["image_name"].map(lambda value: str(image_dir / value))
    missing = [path for path in frame["image_path"] if not Path(path).is_file()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"{split_name}: {len(missing)} images referenced by labels.csv are missing.\n{preview}"
        )
    return frame


def validate_dataset(config: Dict[str, Any], report_path: Optional[Path] = None) -> Dict[str, Any]:
    frames: Dict[str, pd.DataFrame] = {}
    report: Dict[str, Any] = {"splits": {}, "overlap": {}}
    for split in ("train", "val", "test"):
        csv_path, image_dir = resolve_split(config["paths"][f"{split}_dir"])
        frame = read_labels(csv_path, image_dir, split)
        frames[split] = frame
        report["splits"][split] = {
            "count": int(len(frame)),
            "csv": str(csv_path),
            "image_dir": str(image_dir),
            "mos_min": float(frame["mos"].min()),
            "mos_max": float(frame["mos"].max()),
            "mos_mean": float(frame["mos"].mean()),
            "mos_std": float(frame["mos"].std(ddof=0)),
        }

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_keyed = {
            str(row.image_name).replace("\\", "/").lower(): str(row.image_path)
            for row in frames[left].itertuples()
        }
        right_keyed = {
            str(row.image_name).replace("\\", "/").lower(): str(row.image_path)
            for row in frames[right].itertuples()
        }
        left_names = set(left_keyed)
        right_names = set(right_keyed)
        overlap = sorted(left_names.intersection(right_names))
        identical_content = []
        for name in overlap:
            if _sha256(Path(left_keyed[name])) == _sha256(Path(right_keyed[name])):
                identical_content.append(name)
        report["overlap"][f"{left}_{right}"] = {
            "same_relative_name_count": len(overlap),
            "same_relative_name_examples": overlap[:10],
            "identical_content_count": len(identical_content),
            "identical_content_examples": identical_content[:10],
        }
        if identical_content:
            raise ValueError(
                f"Dataset leakage detected between {left} and {right}: "
                f"{len(identical_content)} files have both the same relative name and identical bytes; "
                f"examples={identical_content[:5]}"
            )

    checkpoint = Path(config["paths"]["pretrained_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint does not exist: {checkpoint}")
    report["pretrained_checkpoint"] = str(checkpoint)
    report["status"] = "passed"
    if report_path is not None:
        save_json(report, report_path)
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MOSDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, train: bool, image_size: int = 112):
        self.frame = frame.reset_index(drop=True).copy()
        operations: List[Any] = [transforms.Resize((image_size, image_size))]
        if train:
            # Horizontal flipping preserves face-image quality; blur/color jitter do not.
            operations.append(transforms.RandomHorizontalFlip(p=0.5))
        operations.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.transform = transforms.Compose(operations)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        row = self.frame.iloc[index]
        image_path = Path(row["image_path"])
        try:
            with Image.open(image_path) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {image_path}") from exc
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite pixels after preprocessing: {image_path}")
        return (
            tensor,
            torch.tensor(float(row["mos"]), dtype=torch.float32),
            str(row["image_name"]),
            str(image_path),
        )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, dict) and candidate:
                payload = candidate
                break
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Checkpoint does not contain a non-empty state dict")
    result: Dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        clean = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "backbone.", "model."):
                if clean.startswith(prefix):
                    clean = clean[len(prefix) :]
                    changed = True
        result[clean] = value
    return result


def torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class FIQARegressor(nn.Module):
    """Quality-only wrapper that avoids computing the unused recognition head."""

    def __init__(
        self,
        backbone: nn.Module,
        experiment: str,
        min_log_var: float,
        max_log_var: float,
    ):
        super().__init__()
        self.backbone = backbone
        self.experiment = experiment
        self.min_log_var = float(min_log_var)
        self.max_log_var = float(max_log_var)
        if experiment == "homoscedastic_nll":
            self.global_raw_log_var = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("global_raw_log_var", None)

    def effective_log_var(self, raw: torch.Tensor) -> torch.Tensor:
        # Smooth bounds avoid the zero-gradient dead zones caused by torch.clamp.
        span = self.max_log_var - self.min_log_var
        return self.min_log_var + span * torch.sigmoid(raw)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        tokens = self.backbone.forward_features(images)
        quality_token = tokens[:, 0]
        output = self.backbone.head_qs(quality_token)
        if self.experiment in ("heteroscedastic_nll", "dual_mse_check"):
            mu = output[:, 0]
            if self.experiment == "dual_mse_check":
                return mu, None
            return mu, self.effective_log_var(output[:, 1])
        mu = output[:, 0]
        if self.experiment == "homoscedastic_nll":
            expanded = self.global_raw_log_var.expand_as(mu)
            return mu, self.effective_log_var(expanded)
        return mu, None


def build_model(
    project_root: Path,
    config: Dict[str, Any],
    experiment: str,
    finetune_scope: str,
) -> Tuple[FIQARegressor, Dict[str, Any]]:
    if experiment not in {
        "mse",
        "homoscedastic_nll",
        "heteroscedastic_nll",
        "dual_mse_check",
    }:
        raise ValueError(f"Unsupported experiment: {experiment}")
    if finetune_scope not in {"head_only", "top1", "top2", "top4", "full"}:
        raise ValueError(f"Unsupported fine-tuning scope: {finetune_scope}")

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from backbones.vit_qs import VisionTransformer

    model_cfg = config["model"]
    dual_output = experiment in {"heteroscedastic_nll", "dual_mse_check"}
    backbone = VisionTransformer(
        img_size=int(model_cfg["image_size"]),
        patch_size=int(model_cfg["patch_size"]),
        num_classes=int(model_cfg["embed_dim"]),
        embed_dim=int(model_cfg["embed_dim"]),
        depth=int(model_cfg["depth"]),
        num_heads=int(model_cfg["num_heads"]),
        drop_path_rate=float(model_cfg["drop_path_rate"]),
        norm_layer="ln",
        mask_ratio=float(model_cfg.get("mask_ratio", 0.0)),
        mode="token",
        uncertainty_aware=dual_output,
    )

    checkpoint_path = Path(config["paths"]["pretrained_checkpoint"])
    source = _extract_state_dict(torch_load(checkpoint_path, "cpu"))
    target = backbone.state_dict()
    loaded: Dict[str, torch.Tensor] = {}
    skipped: Dict[str, str] = {}

    for key, value in source.items():
        if key == "head_qs.1.weight":
            continue
        if key in target and tuple(value.shape) == tuple(target[key].shape):
            loaded[key] = value
        elif key in target:
            skipped[key] = f"shape {tuple(value.shape)} -> {tuple(target[key].shape)}"

    pretrained_head = source.get("head_qs.1.weight")
    if pretrained_head is None:
        raise KeyError(
            "Pretrained checkpoint is missing head_qs.1.weight; fair mean-head "
            "initialization cannot be guaranteed"
        )
    if pretrained_head.ndim != 2 or pretrained_head.shape[0] < 1:
        raise ValueError(f"Unexpected pretrained quality-head shape: {tuple(pretrained_head.shape)}")
    if dual_output:
        dual_weight = target["head_qs.1.weight"].clone()
        if pretrained_head.shape[1] != dual_weight.shape[1]:
            raise ValueError("Pretrained and dual quality heads have incompatible input dimensions")
        dual_weight[0].copy_(pretrained_head[0])
        dual_weight[1].zero_()
        loaded["head_qs.1.weight"] = dual_weight
    else:
        if tuple(pretrained_head.shape) != tuple(target["head_qs.1.weight"].shape):
            raise ValueError(
                "Pretrained single-output head shape does not match model: "
                f"{tuple(pretrained_head.shape)} vs {tuple(target['head_qs.1.weight'].shape)}"
            )
        loaded["head_qs.1.weight"] = pretrained_head

    missing, unexpected = backbone.load_state_dict(loaded, strict=False)
    required_missing = [
        key for key in missing if key != "mask_token" and not key.startswith("feature.")
    ]
    if required_missing:
        raise RuntimeError(f"Required pretrained parameters were not loaded: {required_missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading pretrained model: {unexpected}")

    model = FIQARegressor(
        backbone=backbone,
        experiment=experiment,
        min_log_var=float(model_cfg["min_log_var"]),
        max_log_var=float(model_cfg["max_log_var"]),
    )
    configure_trainable_parameters(model, finetune_scope)
    audit = parameter_audit(model, finetune_scope)
    audit.update(
        {
            "checkpoint": str(checkpoint_path),
            "loaded_tensor_count": len(loaded),
            "skipped_shape_mismatches": skipped,
            "dual_output": dual_output,
            "mean_head_initialized_from_pretrained": True,
            "log_variance_head_initialized_to_zero": dual_output,
            "mask_ratio": float(model_cfg.get("mask_ratio", 0.0)),
        }
    )
    return model, audit


def configure_trainable_parameters(model: FIQARegressor, scope: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for parameter in model.backbone.head_qs.parameters():
        parameter.requires_grad_(True)
    if model.global_raw_log_var is not None:
        model.global_raw_log_var.requires_grad_(True)

    if scope == "head_only":
        return
    if scope == "full":
        for name, parameter in model.backbone.named_parameters():
            # The recognition feature head is not in the quality forward path.
            if not name.startswith("feature."):
                parameter.requires_grad_(True)
        return

    block_count = {"top1": 1, "top2": 2, "top4": 4}[scope]
    for block in model.backbone.blocks[-block_count:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    for parameter in model.backbone.norm.parameters():
        parameter.requires_grad_(True)


def parameter_audit(model: FIQARegressor, scope: str) -> Dict[str, Any]:
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters")
    unexpected: List[str] = []
    for name, _ in trainable:
        if name.startswith("backbone.feature."):
            unexpected.append(name)
        if scope == "head_only" and not (
            name.startswith("backbone.head_qs.") or name == "global_raw_log_var"
        ):
            unexpected.append(name)
        if scope.startswith("top") and ".mlp.fc" in name:
            block_index = int(name.split("backbone.blocks.")[1].split(".")[0])
            allowed = int(model.backbone.depth if hasattr(model.backbone, "depth") else len(model.backbone.blocks))
            keep = int(scope.replace("top", ""))
            if block_index < allowed - keep:
                unexpected.append(name)
    if unexpected:
        raise AssertionError(f"Unexpected trainable parameters: {unexpected}")

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    return {
        "scope": scope,
        "total_parameters": int(total),
        "trainable_parameters": int(trainable_count),
        "trainable_ratio": float(trainable_count / total),
        "trainable_names": [name for name, _ in trainable],
    }


def split_optimizer_parameters(model: FIQARegressor) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    head: List[nn.Parameter] = []
    backbone: List[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone.head_qs.") or name == "global_raw_log_var":
            head.append(parameter)
        else:
            backbone.append(parameter)
    return backbone, head


def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * (log_var + (target - mu).square() * torch.exp(-log_var)).mean()


def regression_metrics(
    labels: Sequence[float], predictions: Sequence[float]
) -> Dict[str, float]:
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(predictions, dtype=np.float64)
    if len(y) < 2 or len(y) != len(p):
        raise ValueError("Metrics require equal arrays with at least two samples")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics received non-finite values")
    srcc = float(spearmanr(y, p).statistic)
    plcc = float(pearsonr(y, p).statistic)
    error = p - y
    return {
        "srcc": srcc,
        "plcc": plcc,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
    }


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    scope: str
    seed: int
    role: str = "main"

    @property
    def run_name(self) -> str:
        return f"{self.role}__{self.experiment}__{self.scope}__seed{self.seed}"


def elapsed_text(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def system_information() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info
