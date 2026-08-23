from __future__ import annotations

import gc
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .common import LOGGER, build_model, save_json, torch_load


class ImagePathDataset(Dataset):
    def __init__(self, paths: Sequence[Path], image_size: int):
        self.paths = list(paths)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Failed to read external image: {path}") from exc
        return tensor, str(path)


def _load_heteroscedastic_model(
    project_root: Path,
    run_root: Path,
    config: Dict[str, Any],
    device: torch.device,
):
    seed = int(config["training"]["seeds"][0])
    run_name = f"main__heteroscedastic_nll__top2__seed{seed}"
    checkpoint_path = run_root / "runs" / run_name / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Heteroscedastic checkpoint not found: {checkpoint_path}")
    model, _ = build_model(project_root, config, "heteroscedastic_nll", "top2")
    checkpoint = torch_load(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def run_lfw_analysis(
    project_root: Path,
    run_root: Path,
    config: Dict[str, Any],
) -> None:
    output_dir = run_root / "external_analysis" / "lfw"
    output_dir.mkdir(parents=True, exist_ok=True)
    lfw_dir = Path(config["paths"].get("lfw_dir", ""))
    if not bool(config["analysis"].get("run_lfw_if_available", True)):
        save_json({"status": "disabled"}, output_dir / "status.json")
        return
    if not lfw_dir.is_dir():
        LOGGER.warning("LFW directory is unavailable; skipping optional LFW analysis: %s", lfw_dir)
        save_json(
            {"status": "skipped", "reason": "lfw_dir_not_found", "path": str(lfw_dir)},
            output_dir / "status.json",
        )
        return
    paths = sorted(
        path for path in lfw_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        save_json(
            {"status": "skipped", "reason": "no_images", "path": str(lfw_dir)},
            output_dir / "status.json",
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_heteroscedastic_model(project_root, run_root, config, device)
    dataset = ImagePathDataset(paths, int(config["model"]["image_size"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    records: List[Dict[str, Any]] = []
    amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    with torch.inference_mode():
        for images, batch_paths in loader:
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                mu, log_var = model(images)
            variance = torch.exp(log_var)
            for index, path_text in enumerate(batch_paths):
                path = Path(path_text)
                records.append(
                    {
                        "image_path": str(path),
                        "person_name": path.parent.name,
                        "quality": float(mu[index].float().cpu()),
                        "log_var": float(log_var[index].float().cpu()),
                        "variance": float(variance[index].float().cpu()),
                    }
                )
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "lfw_predictions.csv", index=False, encoding="utf-8-sig")
    correlation = pearsonr(frame["quality"], frame["variance"])
    rank_correlation = spearmanr(frame["quality"], frame["variance"])
    summary = {
        "status": "completed",
        "sample_count": int(len(frame)),
        "quality_mean": float(frame["quality"].mean()),
        "quality_std": float(frame["quality"].std(ddof=1)),
        "variance_mean": float(frame["variance"].mean()),
        "variance_std": float(frame["variance"].std(ddof=1)),
        "quality_variance_pearson": float(correlation.statistic),
        "quality_variance_pearson_p": float(correlation.pvalue),
        "quality_variance_spearman": float(rank_correlation.statistic),
        "quality_variance_spearman_p": float(rank_correlation.pvalue),
        "note": "LFW has no MOS; these are descriptive out-of-domain results, not quantitative FIQA accuracy.",
    }
    save_json(summary, output_dir / "lfw_summary.json")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    sns.histplot(frame["quality"], bins=50, kde=True, ax=axes[0], color="#4C78A8")
    axes[0].set(title="LFW predicted quality", xlabel="Quality")
    sns.histplot(frame["variance"], bins=50, kde=True, ax=axes[1], color="#54A24B")
    axes[1].set(title="LFW predictive variance", xlabel="Variance")
    axes[2].hexbin(frame["quality"], frame["variance"], gridsize=45, mincnt=1, cmap="magma")
    axes[2].set(
        title=f"Quality vs. variance (r={summary['quality_variance_pearson']:.3f})",
        xlabel="Quality",
        ylabel="Variance",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "lfw_distribution_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    groups = [
        ("Highest quality", frame.nlargest(5, "quality")),
        ("Highest uncertainty", frame.nlargest(5, "variance")),
        ("Lowest quality", frame.nsmallest(5, "quality")),
    ]
    fig, axes = plt.subplots(3, 5, figsize=(14, 8.2))
    for row_index, (label, group) in enumerate(groups):
        for column_index, (_, item) in enumerate(group.iterrows()):
            with Image.open(item["image_path"]) as image:
                axes[row_index, column_index].imshow(image.convert("RGB"))
            axes[row_index, column_index].set_title(
                f"Q={item['quality']:.3f}\nU={item['variance']:.4f}", fontsize=9
            )
            axes[row_index, column_index].axis("off")
        axes[row_index, 0].set_ylabel(label, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "lfw_representative_samples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    LOGGER.info("Completed optional LFW descriptive analysis on %d images", len(frame))


def _apply_degradation(image: Image.Image, kind: str, level: float) -> Image.Image:
    if kind == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=level))
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(level)
    if kind == "downsample":
        width, height = image.size
        small = image.resize(
            (max(4, int(width * level)), max(4, int(height * level))),
            Image.Resampling.BILINEAR,
        )
        return small.resize((width, height), Image.Resampling.BILINEAR)
    if kind == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(level))
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()
    raise ValueError(f"Unknown degradation kind: {kind}")


def run_controlled_degradation_analysis(
    project_root: Path,
    run_root: Path,
    config: Dict[str, Any],
) -> None:
    output_dir = run_root / "external_analysis" / "controlled_degradation"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["training"]["seeds"][0])
    prediction_path = (
        run_root
        / "runs"
        / f"main__heteroscedastic_nll__top2__seed{seed}"
        / "test_predictions.csv"
    )
    predictions = pd.read_csv(prediction_path, encoding="utf-8-sig")
    sample_count = min(int(config["analysis"].get("degradation_samples", 60)), len(predictions))
    # Use clear images so each synthetic degradation starts from a meaningful reference.
    selected = predictions.nlargest(sample_count, "mos")
    settings = {
        "gaussian_blur": [0.0, 0.75, 1.5, 3.0],
        "brightness": [1.0, 0.75, 0.5, 0.25],
        "downsample": [1.0, 0.75, 0.5, 0.25],
        "jpeg": [95.0, 70.0, 40.0, 15.0],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_heteroscedastic_model(project_root, run_root, config, device)
    image_size = int(config["model"]["image_size"])
    normalize = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    records: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for kind, levels in settings.items():
            for level_index, level in enumerate(levels):
                tensors = []
                names = []
                for _, row in selected.iterrows():
                    with Image.open(row["image_path"]) as image:
                        rgb = image.convert("RGB")
                        degraded = _apply_degradation(rgb, kind, level)
                    tensors.append(normalize(degraded))
                    names.append(str(row["image_name"]))
                for start in range(0, len(tensors), int(config["training"]["batch_size"])):
                    batch = torch.stack(tensors[start : start + int(config["training"]["batch_size"])]).to(device)
                    with torch.cuda.amp.autocast(enabled=amp):
                        mu, log_var = model(batch)
                    variance = torch.exp(log_var)
                    for local_index in range(len(batch)):
                        global_index = start + local_index
                        records.append(
                            {
                                "image_name": names[global_index],
                                "degradation": kind,
                                "level_index": level_index,
                                "level_value": float(level),
                                "quality": float(mu[local_index].float().cpu()),
                                "variance": float(variance[local_index].float().cpu()),
                            }
                        )
    frame = pd.DataFrame(records)
    frame.to_csv(output_dir / "controlled_degradation_predictions.csv", index=False, encoding="utf-8-sig")
    aggregate = (
        frame.groupby(["degradation", "level_index", "level_value"])[["quality", "variance"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.columns = ["_".join(filter(None, map(str, column))).rstrip("_") for column in aggregate.columns]
    aggregate.to_csv(output_dir / "controlled_degradation_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, kind in zip(axes.ravel(), settings):
        subset = aggregate[aggregate["degradation"] == kind]
        x = subset["level_index"]
        ax.errorbar(x, subset["quality_mean"], yerr=subset["quality_std"], marker="o", label="Quality")
        twin = ax.twinx()
        twin.errorbar(x, subset["variance_mean"], yerr=subset["variance_std"], marker="s", color="#E45756", label="Variance")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{value:g}" for value in subset["level_value"]])
        ax.set(title=kind.replace("_", " ").title(), xlabel="Degradation parameter", ylabel="Mean quality")
        twin.set_ylabel("Mean variance", color="#E45756")
    fig.suptitle("Controlled degradation sensitivity (mean +/- SD)")
    fig.tight_layout()
    fig.savefig(output_dir / "controlled_degradation_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    save_json(
        {
            "status": "completed",
            "source_test_images": int(sample_count),
            "prediction_count": int(len(frame)),
            "note": "This is a controlled sensitivity test; it does not replace MOS-based test metrics.",
        },
        output_dir / "status.json",
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    LOGGER.info("Completed controlled degradation analysis")
