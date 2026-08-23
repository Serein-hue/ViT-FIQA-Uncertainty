from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm, pearsonr, spearmanr, wilcoxon

from .common import LOGGER, RunSpec, regression_metrics, save_json


sns.set_theme(style="whitegrid", context="paper")
COLORS = {
    "mse": "#4C78A8",
    "homoscedastic_nll": "#F58518",
    "heteroscedastic_nll": "#54A24B",
    "dual_mse_check": "#B279A2",
}


def _safe_corr(left: np.ndarray, right: np.ndarray) -> Dict[str, Optional[float]]:
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return {
            "spearman": None,
            "spearman_p": None,
            "pearson": None,
            "pearson_p": None,
        }
    sr = spearmanr(left, right)
    pr = pearsonr(left, right)
    return {
        "spearman": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
        "pearson": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
    }


def plot_training_history(run_dir: Path) -> None:
    frame = pd.read_csv(run_dir / "training_history.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(frame["epoch"], frame["train_loss"], marker="o", markersize=3)
    axes[0, 0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    axes[0, 1].plot(frame["epoch"], frame["val_srcc"], label="SRCC")
    axes[0, 1].plot(frame["epoch"], frame["val_plcc"], label="PLCC")
    axes[0, 1].set(title="Validation correlation", xlabel="Epoch", ylabel="Correlation")
    axes[0, 1].legend()
    axes[1, 0].plot(frame["epoch"], frame["val_mae"], label="MAE")
    axes[1, 0].plot(frame["epoch"], frame["val_rmse"], label="RMSE")
    axes[1, 0].set(title="Validation error", xlabel="Epoch", ylabel="Error")
    axes[1, 0].legend()
    axes[1, 1].plot(frame["epoch"], frame["backbone_lr"], label="Backbone LR")
    axes[1, 1].plot(frame["epoch"], frame["head_lr"], label="Head LR")
    axes[1, 1].set(title="Learning-rate schedule", xlabel="Epoch", ylabel="Learning rate")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_scatter(frame: pd.DataFrame, run_dir: Path) -> None:
    metrics = regression_metrics(frame["mos"], frame["mu"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.hexbin(frame["mos"], frame["mu"], gridsize=45, mincnt=1, cmap="viridis")
    low = float(min(frame["mos"].min(), frame["mu"].min()))
    high = float(max(frame["mos"].max(), frame["mu"].max()))
    ax.plot([low, high], [low, high], "r--", linewidth=1.5, label="Ideal")
    ax.set(
        title=f"Prediction vs. MOS (SRCC={metrics['srcc']:.3f}, PLCC={metrics['plcc']:.3f})",
        xlabel="Ground-truth MOS",
        ylabel="Predicted quality",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "prediction_vs_mos.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _risk_coverage(
    frame: pd.DataFrame, coverages: Sequence[float], random_repeats: int = 100
) -> pd.DataFrame:
    error = frame["abs_error"].to_numpy(dtype=float)
    uncertainty = frame["variance"].to_numpy(dtype=float)
    n = len(frame)
    uncertainty_order = np.argsort(uncertainty)
    oracle_order = np.argsort(error)
    rng = np.random.default_rng(20260731)
    records = []
    for coverage in coverages:
        keep = max(2, min(n, int(round(n * float(coverage)))))
        random_risks = []
        for _ in range(random_repeats):
            chosen = rng.choice(n, size=keep, replace=False)
            random_risks.append(float(error[chosen].mean()))
        records.append(
            {
                "coverage": float(keep / n),
                "rejection_rate": float(1.0 - keep / n),
                "uncertainty_mae": float(error[uncertainty_order[:keep]].mean()),
                "oracle_mae": float(error[oracle_order[:keep]].mean()),
                "random_mae": float(np.mean(random_risks)),
                "random_mae_std": float(np.std(random_risks, ddof=1)),
            }
        )
    return pd.DataFrame(records).sort_values("coverage", ascending=False).reset_index(drop=True)


def _prediction_intervals(
    frame: pd.DataFrame, levels: Sequence[float]
) -> pd.DataFrame:
    y = frame["mos"].to_numpy(dtype=float)
    mu = frame["mu"].to_numpy(dtype=float)
    std = frame["std"].to_numpy(dtype=float)
    records = []
    for level in levels:
        z = float(norm.ppf((1.0 + float(level)) / 2.0))
        lower = mu - z * std
        upper = mu + z * std
        covered = (y >= lower) & (y <= upper)
        records.append(
            {
                "nominal_coverage": float(level),
                "empirical_coverage": float(covered.mean()),
                "coverage_gap": float(covered.mean() - level),
                "mean_interval_width": float(np.mean(upper - lower)),
            }
        )
    return pd.DataFrame(records)


def _uncertainty_bins(frame: pd.DataFrame, bin_count: int) -> pd.DataFrame:
    working = frame.copy()
    working["uncertainty_bin"] = pd.qcut(
        working["variance"], q=bin_count, labels=False, duplicates="drop"
    )
    records = []
    for bin_id, group in working.groupby("uncertainty_bin", observed=True):
        rmse = float(np.sqrt(group["squared_error"].mean()))
        root_mean_variance = float(np.sqrt(group["variance"].mean()))
        records.append(
            {
                "bin": int(bin_id) + 1,
                "count": int(len(group)),
                "mean_variance": float(group["variance"].mean()),
                "mean_std": float(group["std"].mean()),
                "mae": float(group["abs_error"].mean()),
                "rmse": rmse,
                "root_mean_variance": root_mean_variance,
                "calibration_gap": root_mean_variance - rmse,
            }
        )
    return pd.DataFrame(records)


def uncertainty_analysis(
    frame: pd.DataFrame,
    run_dir: Path,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    if not frame["variance"].notna().all():
        raise ValueError("Uncertainty analysis requires a variance for every sample")
    analysis_cfg = config["analysis"]
    variance = frame["variance"].to_numpy(dtype=float)
    abs_error = frame["abs_error"].to_numpy(dtype=float)
    squared_error = frame["squared_error"].to_numpy(dtype=float)
    residual = frame["mos"].to_numpy(dtype=float) - frame["mu"].to_numpy(dtype=float)
    standardized = residual / np.maximum(frame["std"].to_numpy(dtype=float), 1e-12)

    correlation = _safe_corr(variance, abs_error)
    correlation_label = (
        "NA" if correlation["spearman"] is None else f"{correlation['spearman']:.3f}"
    )
    bins = _uncertainty_bins(frame, int(analysis_cfg["uncertainty_bins"]))
    bins.to_csv(run_dir / "uncertainty_quintiles.csv", index=False, encoding="utf-8-sig")
    risk = _risk_coverage(frame, analysis_cfg["risk_coverages"])
    risk.to_csv(run_dir / "risk_coverage.csv", index=False, encoding="utf-8-sig")
    intervals = _prediction_intervals(frame, analysis_cfg["prediction_interval_levels"])
    intervals.to_csv(run_dir / "prediction_interval_calibration.csv", index=False, encoding="utf-8-sig")

    ence_terms = np.abs(bins["root_mean_variance"] - bins["rmse"]) / np.maximum(
        bins["root_mean_variance"], 1e-12
    )
    coverage_ascending = risk.sort_values("coverage")
    aurc = float(
        np.trapz(
            coverage_ascending["uncertainty_mae"],
            coverage_ascending["coverage"],
        )
    )
    ause = float(
        np.trapz(
            coverage_ascending["uncertainty_mae"] - coverage_ascending["oracle_mae"],
            coverage_ascending["coverage"],
        )
    )
    summary: Dict[str, Any] = {
        "uncertainty_abs_error_correlation": correlation,
        "mean_variance": float(np.mean(variance)),
        "std_variance": float(np.std(variance, ddof=1)),
        "variance_min": float(np.min(variance)),
        "variance_max": float(np.max(variance)),
        "gaussian_nll": float(np.mean(0.5 * (np.log(variance) + squared_error / variance))),
        "ence": float(np.mean(ence_terms)),
        "aurc_mae": aurc,
        "ause_mae": ause,
        "standardized_residual_mean": float(np.mean(standardized)),
        "standardized_residual_std": float(np.std(standardized, ddof=1)),
    }
    save_json(summary, run_dir / "uncertainty_summary.json")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    sns.histplot(frame["variance"], bins=40, kde=True, ax=axes[0, 0], color="#54A24B")
    axes[0, 0].set(title="Predictive variance distribution", xlabel="Variance")
    hb = axes[0, 1].hexbin(variance, abs_error, gridsize=45, mincnt=1, cmap="magma")
    axes[0, 1].set(
        title=f"Uncertainty vs. absolute error (SRCC={correlation_label})",
        xlabel="Predictive variance",
        ylabel="Absolute error",
    )
    fig.colorbar(hb, ax=axes[0, 1], label="Samples per hexagon")
    axes[1, 0].plot(bins["bin"], bins["mae"], marker="o", label="MAE")
    axes[1, 0].plot(bins["bin"], bins["rmse"], marker="s", label="RMSE")
    axes[1, 0].set(
        title="Error by uncertainty quantile",
        xlabel="Uncertainty bin (low to high)",
        ylabel="Error",
    )
    axes[1, 0].legend()
    axes[1, 1].plot(risk["rejection_rate"], risk["uncertainty_mae"], marker="o", label="Uncertainty")
    axes[1, 1].plot(risk["rejection_rate"], risk["random_mae"], marker="s", label="Random")
    axes[1, 1].plot(risk["rejection_rate"], risk["oracle_mae"], linestyle="--", label="Oracle")
    axes[1, 1].set(
        title="Selective prediction / rejection curve",
        xlabel="Rejected fraction",
        ylabel="MAE of retained samples",
    )
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(run_dir / "uncertainty_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(intervals["nominal_coverage"], intervals["empirical_coverage"], marker="o")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set(
        title="Prediction-interval calibration",
        xlabel="Nominal coverage",
        ylabel="Empirical coverage",
        xlim=(0.45, 1.0),
        ylim=(0.45, 1.0),
    )
    axes[1].plot(bins["root_mean_variance"], bins["rmse"], marker="o")
    max_axis = float(max(bins["root_mean_variance"].max(), bins["rmse"].max()))
    axes[1].plot([0, max_axis], [0, max_axis], "k--", linewidth=1)
    axes[1].set(
        title=f"RMV-RMSE reliability (ENCE={summary['ence']:.3f})",
        xlabel="Root mean variance",
        ylabel="RMSE",
    )
    sns.histplot(standardized, bins=40, stat="density", ax=axes[2], color="#E45756")
    grid = np.linspace(-4, 4, 300)
    axes[2].plot(grid, norm.pdf(grid), "k--", label="N(0,1)")
    axes[2].set(title="Standardized residuals", xlabel="(MOS - mu) / predicted std")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(run_dir / "uncertainty_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary


def analyze_run(run_dir: Path, config: Dict[str, Any]) -> None:
    if not (run_dir / "completed.json").is_file():
        return
    plot_training_history(run_dir)
    predictions = pd.read_csv(run_dir / "test_predictions.csv", encoding="utf-8-sig")
    plot_prediction_scatter(predictions, run_dir)
    if predictions["variance"].notna().all() and predictions["variance"].nunique() > 2:
        uncertainty_analysis(predictions, run_dir, config)


def _flatten_result(result: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        key: result[key]
        for key in (
            "run_name",
            "experiment",
            "scope",
            "seed",
            "role",
            "best_epoch",
            "epochs_ran",
            "train_seconds",
            "peak_gpu_memory_mb",
            "total_parameters",
            "trainable_parameters",
            "trainable_ratio",
        )
    }
    for prefix in ("val", "test"):
        for key, value in result[prefix].items():
            row[f"{prefix}_{key}"] = value
    return row


def _paired_bootstrap(
    baseline: pd.DataFrame,
    proposed: pd.DataFrame,
    samples: int,
    seed: int,
) -> Dict[str, float]:
    merged = baseline[["image_name", "mos", "mu"]].merge(
        proposed[["image_name", "mos", "mu"]],
        on=["image_name", "mos"],
        suffixes=("_baseline", "_proposed"),
        validate="one_to_one",
    )
    n = len(merged)
    rng = np.random.default_rng(seed)
    mae_differences = []
    srcc_differences = []
    y = merged["mos"].to_numpy()
    baseline_mu = merged["mu_baseline"].to_numpy()
    proposed_mu = merged["mu_proposed"].to_numpy()
    for _ in range(samples):
        index = rng.integers(0, n, size=n)
        y_b = y[index]
        base_b = baseline_mu[index]
        prop_b = proposed_mu[index]
        mae_differences.append(
            float(np.mean(np.abs(prop_b - y_b)) - np.mean(np.abs(base_b - y_b)))
        )
        if np.std(y_b) > 0 and np.std(base_b) > 0 and np.std(prop_b) > 0:
            srcc_differences.append(
                float(spearmanr(y_b, prop_b).statistic - spearmanr(y_b, base_b).statistic)
            )
    return {
        "mae_difference_proposed_minus_baseline": float(np.mean(mae_differences)),
        "mae_difference_ci_low": float(np.quantile(mae_differences, 0.025)),
        "mae_difference_ci_high": float(np.quantile(mae_differences, 0.975)),
        "srcc_difference_proposed_minus_baseline": float(np.mean(srcc_differences)),
        "srcc_difference_ci_low": float(np.quantile(srcc_differences, 0.025)),
        "srcc_difference_ci_high": float(np.quantile(srcc_differences, 0.975)),
    }


def statistical_comparison(
    run_root: Path,
    summary: pd.DataFrame,
    bootstrap_samples: int,
) -> pd.DataFrame:
    records = []
    main = summary[(summary["role"] == "main") & (summary["scope"] == "top2")]
    proposed_seeds = set(main.loc[main["experiment"] == "heteroscedastic_nll", "seed"])
    for baseline_experiment in ("mse", "homoscedastic_nll"):
        seeds = sorted(
            set(main.loc[main["experiment"] == baseline_experiment, "seed"]).intersection(
                proposed_seeds
            )
        )
        for seed in seeds:
            base_row = main[
                (main["experiment"] == baseline_experiment) & (main["seed"] == seed)
            ].iloc[0]
            prop_row = main[
                (main["experiment"] == "heteroscedastic_nll") & (main["seed"] == seed)
            ].iloc[0]
            baseline = pd.read_csv(
                run_root / str(base_row["run_name"]) / "test_predictions.csv", encoding="utf-8-sig"
            )
            proposed = pd.read_csv(
                run_root / str(prop_row["run_name"]) / "test_predictions.csv", encoding="utf-8-sig"
            )
            merged = baseline[["image_name", "abs_error"]].merge(
                proposed[["image_name", "abs_error"]],
                on="image_name",
                suffixes=("_baseline", "_proposed"),
                validate="one_to_one",
            )
            try:
                test = wilcoxon(
                    merged["abs_error_proposed"],
                    merged["abs_error_baseline"],
                    alternative="two-sided",
                    zero_method="wilcox",
                )
                statistic = float(test.statistic)
                p_value = float(test.pvalue)
            except ValueError:
                statistic = 0.0
                p_value = 1.0
            record = {
                "baseline_experiment": baseline_experiment,
                "proposed_experiment": "heteroscedastic_nll",
                "seed": int(seed),
                "sample_count": int(len(merged)),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p": p_value,
            }
            record.update(
                _paired_bootstrap(
                    baseline,
                    proposed,
                    samples=bootstrap_samples,
                    seed=9000 + int(seed) + (0 if baseline_experiment == "mse" else 100),
                )
            )
            records.append(record)
    return pd.DataFrame(records)


def aggregate_results(run_root: Path, config: Dict[str, Any]) -> pd.DataFrame:
    results: List[Dict[str, Any]] = []
    for completed in sorted((run_root / "runs").glob("*/completed.json")):
        with completed.open("r", encoding="utf-8") as handle:
            results.append(json.load(handle))
    if not results:
        raise RuntimeError("No completed experiment runs were found")
    summary = pd.DataFrame([_flatten_result(result) for result in results])
    summary.to_csv(run_root / "all_runs_summary.csv", index=False, encoding="utf-8-sig")

    main = summary[(summary["role"] == "main") & (summary["scope"] == "top2")].copy()
    metric_columns = ["test_srcc", "test_plcc", "test_mae", "test_rmse"]
    aggregate = main.groupby("experiment")[metric_columns].agg(["mean", "std", "count"])
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate.to_csv(
        run_root / "main_experiments_mean_std.csv", index=False, encoding="utf-8-sig"
    )

    order = [name for name in ("mse", "homoscedastic_nll", "heteroscedastic_nll") if name in set(main["experiment"])]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, metric, label in zip(
        axes,
        metric_columns,
        ("SRCC", "PLCC", "MAE", "RMSE"),
    ):
        grouped = main.groupby("experiment")[metric]
        means = grouped.mean().reindex(order)
        stds = grouped.std().fillna(0).reindex(order)
        colors = [COLORS[name] for name in order]
        ax.bar(range(len(order)), means, yerr=stds, capsize=4, color=colors)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([name.replace("_", "\n") for name in order], rotation=0)
        ax.set_title(label)
        if metric in {"test_srcc", "test_plcc"}:
            bottom = max(0.0, float(means.min() - 0.1))
            ax.set_ylim(bottom, 1.0)
    fig.suptitle("Official test-set performance (mean +/- standard deviation across seeds)")
    fig.tight_layout()
    fig.savefig(run_root / "main_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    scope = summary[summary["role"] == "scope"].copy()
    top2_reference = summary[
        (summary["role"] == "main")
        & (summary["experiment"] == "heteroscedastic_nll")
        & (summary["seed"] == int(config["training"]["seeds"][0]))
    ]
    scope = pd.concat([scope, top2_reference], ignore_index=True)
    if not scope.empty:
        scope.to_csv(run_root / "finetune_scope_results.csv", index=False, encoding="utf-8-sig")
        fig, ax = plt.subplots(figsize=(7, 5))
        for _, row in scope.iterrows():
            ax.scatter(row["trainable_parameters"], row["test_srcc"], s=80)
            ax.annotate(row["scope"], (row["trainable_parameters"], row["test_srcc"]), xytext=(5, 5), textcoords="offset points")
        ax.set_xscale("log")
        ax.set(
            title="Fine-tuning efficiency",
            xlabel="Trainable parameters (log scale)",
            ylabel="Test SRCC",
        )
        fig.tight_layout()
        fig.savefig(run_root / "finetune_efficiency.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    statistics = statistical_comparison(
        run_root / "runs", summary, int(config["analysis"]["bootstrap_samples"])
    )
    if not statistics.empty:
        statistics.to_csv(run_root / "paired_statistical_tests.csv", index=False, encoding="utf-8-sig")

    _write_manuscript_summary(run_root, summary, aggregate, statistics)
    return summary


def _format_mean_std(aggregate: pd.DataFrame, experiment: str, metric: str) -> str:
    row = aggregate.loc[experiment]
    mean = float(row[f"{metric}_mean"])
    std = float(row[f"{metric}_std"])
    return f"{mean:.4f} +/- {std:.4f}"


def _write_manuscript_summary(
    run_root: Path,
    summary: pd.DataFrame,
    aggregate: pd.DataFrame,
    statistics: pd.DataFrame,
) -> None:
    aggregate_indexed = aggregate.set_index("experiment")
    lines = [
        "# Paper experiment result summary",
        "",
        "All values below come from the official test split, evaluated once after validation-based model selection.",
        "",
        "| Method | SRCC | PLCC | MAE | RMSE | Seeds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for experiment in ("mse", "homoscedastic_nll", "heteroscedastic_nll"):
        if experiment not in aggregate_indexed.index:
            continue
        row = aggregate_indexed.loc[experiment]
        lines.append(
            "| {name} | {srcc} | {plcc} | {mae} | {rmse} | {count} |".format(
                name=experiment,
                srcc=f"{row['test_srcc_mean']:.4f} +/- {row['test_srcc_std']:.4f}",
                plcc=f"{row['test_plcc_mean']:.4f} +/- {row['test_plcc_std']:.4f}",
                mae=f"{row['test_mae_mean']:.4f} +/- {row['test_mae_std']:.4f}",
                rmse=f"{row['test_rmse_mean']:.4f} +/- {row['test_rmse_std']:.4f}",
                count=int(row["test_srcc_count"]),
            )
        )
    lines.extend(["", "## Statistical comparison", ""])
    if statistics.empty:
        lines.append("Paired MSE-versus-heteroscedastic comparisons were not available.")
    else:
        for _, row in statistics.iterrows():
            lines.append(
                f"- Heteroscedastic vs {row['baseline_experiment']}, seed {int(row['seed'])}: "
                f"Wilcoxon p={row['wilcoxon_p']:.4g}; "
                f"MAE difference (heteroscedastic - MSE) 95% CI "
                f"[{row['mae_difference_ci_low']:.5f}, {row['mae_difference_ci_high']:.5f}]; "
                f"SRCC difference 95% CI "
                f"[{row['srcc_difference_ci_low']:.5f}, {row['srcc_difference_ci_high']:.5f}]."
            )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "Do not claim that the heteroscedastic model is better unless the repeated-seed results, paired error test, and uncertainty diagnostics all support that wording. Report negative or neutral findings as observed.",
        ]
    )
    (run_root / "manuscript_results_summary.md").write_text("\n".join(lines), encoding="utf-8")
