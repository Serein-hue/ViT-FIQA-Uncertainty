from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from .common import LOGGER, build_model, torch_load


def _select_samples(predictions: pd.DataFrame, count: int) -> pd.DataFrame:
    candidates: List[pd.Series] = []
    per_group = max(1, count // 3)
    candidates.extend([row for _, row in predictions.nlargest(per_group, "variance").iterrows()])
    candidates.extend([row for _, row in predictions.nsmallest(per_group, "mu").iterrows()])
    candidates.extend([row for _, row in predictions.nlargest(per_group, "mu").iterrows()])
    selected: Dict[str, pd.Series] = {}
    for row in candidates:
        selected.setdefault(str(row["image_name"]), row)
    if len(selected) < count:
        remaining = predictions.sort_values("variance", ascending=False)
        for _, row in remaining.iterrows():
            selected.setdefault(str(row["image_name"]), row)
            if len(selected) >= count:
                break
    return pd.DataFrame(list(selected.values())[:count]).reset_index(drop=True)


def _preprocess(image_path: str, image_size: int) -> Tuple[torch.Tensor, np.ndarray]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        display = np.asarray(rgb).copy()
    tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )(Image.fromarray(display))
    return tensor.unsqueeze(0), display


def _extract_for_model(
    project_root: Path,
    config: Dict[str, Any],
    experiment: str,
    checkpoint_path: Path,
    selected: pd.DataFrame,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    model, _ = build_model(project_root, config, experiment, "top2")
    checkpoint = torch_load(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    captured: Dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["attention"] = output.detach()

    handle = model.backbone.blocks[-1].attn.attn_drop.register_forward_hook(hook)
    output: Dict[str, Dict[str, Any]] = {}
    image_size = int(config["model"]["image_size"])
    try:
        with torch.inference_mode():
            for _, row in selected.iterrows():
                tensor, display = _preprocess(str(row["image_path"]), image_size)
                captured.clear()
                mu, log_var = model(tensor.to(device))
                if "attention" not in captured:
                    raise RuntimeError("Attention hook did not capture the last-layer attention tensor")
                attention = captured["attention"]
                if attention.ndim != 4:
                    raise RuntimeError(f"Unexpected attention shape: {tuple(attention.shape)}")
                patch_attention = attention[0, :, 0, 1:].mean(dim=0).float().cpu().numpy()
                grid_size = int(round(math.sqrt(len(patch_attention))))
                if grid_size * grid_size != len(patch_attention):
                    raise RuntimeError(
                        f"Patch attention length {len(patch_attention)} is not a square grid"
                    )
                heat = patch_attention.reshape(grid_size, grid_size)
                heat = cv2.resize(heat, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
                heat = heat - float(heat.min())
                denominator = float(heat.max())
                if denominator <= 1e-12:
                    raise RuntimeError("Captured attention map is constant; refusing to create a fake heatmap")
                heat = heat / denominator
                output[str(row["image_name"])] = {
                    "heat": heat,
                    "display": display,
                    "mu": float(mu.item()),
                    "variance": float(torch.exp(log_var).item()) if log_var is not None else math.nan,
                }
    finally:
        handle.remove()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def _overlay(display: np.ndarray, heat: np.ndarray) -> np.ndarray:
    colored = cv2.applyColorMap(np.uint8(np.clip(heat, 0, 1) * 255), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(display, 0.58, colored, 0.42, 0)


def generate_attention_comparison(
    project_root: Path,
    run_root: Path,
    config: Dict[str, Any],
) -> None:
    seed = int(config["training"]["seeds"][0])
    mse_name = f"main__mse__top2__seed{seed}"
    hetero_name = f"main__heteroscedastic_nll__top2__seed{seed}"
    mse_dir = run_root / "runs" / mse_name
    hetero_dir = run_root / "runs" / hetero_name
    if not (mse_dir / "best_model.pt").is_file() or not (hetero_dir / "best_model.pt").is_file():
        LOGGER.warning("Skipping attention comparison because required checkpoints are missing")
        return
    predictions = pd.read_csv(hetero_dir / "test_predictions.csv", encoding="utf-8-sig")
    selected = _select_samples(predictions, int(config["analysis"]["attention_samples"]))
    selected.to_csv(run_root / "attention_selected_samples.csv", index=False, encoding="utf-8-sig")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mse_maps = _extract_for_model(
        project_root, config, "mse", mse_dir / "best_model.pt", selected, device
    )
    hetero_maps = _extract_for_model(
        project_root,
        config,
        "heteroscedastic_nll",
        hetero_dir / "best_model.pt",
        selected,
        device,
    )

    rows = len(selected)
    fig, axes = plt.subplots(rows, 3, figsize=(10, 3.1 * rows), squeeze=False)
    for row_index, (_, row) in enumerate(selected.iterrows()):
        name = str(row["image_name"])
        baseline = mse_maps[name]
        proposed = hetero_maps[name]
        axes[row_index, 0].imshow(proposed["display"])
        axes[row_index, 0].set_title(f"Input\nMOS={float(row['mos']):.3f}")
        axes[row_index, 1].imshow(_overlay(baseline["display"], baseline["heat"]))
        axes[row_index, 1].set_title(f"MSE QS attention\nQ={baseline['mu']:.3f}")
        axes[row_index, 2].imshow(_overlay(proposed["display"], proposed["heat"]))
        axes[row_index, 2].set_title(
            f"Heteroscedastic QS attention\nQ={proposed['mu']:.3f}, U={proposed['variance']:.4f}"
        )
        for column in range(3):
            axes[row_index, column].axis("off")
    fig.suptitle("True last-layer QS-token attention (mean across heads)", y=1.002)
    fig.tight_layout()
    fig.savefig(run_root / "true_qs_attention_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved genuine QS-token attention comparison")
