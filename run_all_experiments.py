from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from paper_experiments.analysis import aggregate_results, analyze_run
from paper_experiments.attention import generate_attention_comparison
from paper_experiments.extra_analysis import (
    run_controlled_degradation_analysis,
    run_lfw_analysis,
)
from paper_experiments.common import (
    LOGGER,
    RunSpec,
    load_json,
    read_labels,
    resolve_split,
    save_json,
    setup_logging,
    system_information,
    validate_dataset,
)
from paper_experiments.train import train_one_run


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "experiment_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete reproducible FIQA paper experiment pipeline."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--profile",
        choices=("core", "standard", "extended"),
        default="standard",
        help="core=3 one-seed models; standard=3x3 seeds plus controls; extended=extra scopes",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Existing timestamped outcome directory to resume; completed runs are skipped",
    )
    parser.add_argument("--check-only", action="store_true", help="Validate paths/data/model file only")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one epoch on 16 samples per split to validate the complete pipeline",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print and save the run plan only")
    return parser.parse_args()


def make_plan(config: Dict[str, Any], profile: str, smoke_test: bool) -> List[RunSpec]:
    seeds = [int(value) for value in config["training"]["seeds"]]
    if not seeds:
        raise ValueError("training.seeds must not be empty")
    main_seeds = seeds[:1] if profile == "core" or smoke_test else seeds
    specs = [
        RunSpec(experiment, "top2", seed, "main")
        for seed in main_seeds
        for experiment in ("mse", "homoscedastic_nll", "heteroscedastic_nll")
    ]
    if profile in {"standard", "extended"} and not smoke_test:
        reference_seed = seeds[0]
        specs.extend(
            [
                RunSpec("dual_mse_check", "top2", reference_seed, "sanity"),
                RunSpec("heteroscedastic_nll", "head_only", reference_seed, "scope"),
                RunSpec("heteroscedastic_nll", "full", reference_seed, "scope"),
            ]
        )
    if profile == "extended" and not smoke_test:
        reference_seed = seeds[0]
        specs.extend(
            [
                RunSpec("heteroscedastic_nll", "top1", reference_seed, "scope"),
                RunSpec("heteroscedastic_nll", "top4", reference_seed, "scope"),
            ]
        )
    names = [spec.run_name for spec in specs]
    if len(names) != len(set(names)):
        raise AssertionError("Experiment plan contains duplicate run names")
    return specs


def prepare_run_root(config: Dict[str, Any], args: argparse.Namespace) -> Path:
    if args.resume is not None:
        run_root = args.resume.resolve()
        if not run_root.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {run_root}")
        return run_root
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "SMOKE" if args.smoke_test else "FIQA_PAPER"
    run_root = Path(config["paths"]["output_root"]) / f"{prefix}_{stamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


def load_split_frames(config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        csv_path, image_dir = resolve_split(config["paths"][f"{split}_dir"])
        frames[split] = read_labels(csv_path, image_dir, split)
    return frames


def snapshot_sources(run_root: Path, config_path: Path) -> None:
    snapshot = run_root / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, snapshot / "experiment_config.json")
    shutil.copy2(PROJECT_ROOT / "run_all_experiments.py", snapshot / "run_all_experiments.py")
    target_package = snapshot / "paper_experiments"
    if target_package.exists():
        shutil.rmtree(target_package)
    shutil.copytree(
        PROJECT_ROOT / "paper_experiments",
        target_package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(PROJECT_ROOT / "backbones" / "vit_qs.py", snapshot / "vit_qs.py")


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    run_root = prepare_run_root(config, args)
    setup_logging(run_root / "pipeline.log")
    LOGGER.info("Outcome directory: %s", run_root)
    prior_config_path = run_root / "resolved_config.json"
    if args.resume is not None and prior_config_path.is_file():
        prior_config = load_json(prior_config_path)
        if prior_config != config:
            raise ValueError(
                "The current experiment_config.json differs from the configuration stored in the "
                "resume directory. Start a new outcome directory to avoid mixing incompatible runs."
            )
    prior_plan_path = run_root / "experiment_plan.json"
    if args.resume is not None and prior_plan_path.is_file():
        prior_plan = load_json(prior_plan_path)
        if prior_plan.get("profile") != args.profile or bool(prior_plan.get("smoke_test")) != bool(args.smoke_test):
            raise ValueError("Resume profile/smoke-test mode does not match the original experiment plan")
    save_json(system_information(), run_root / "system_info.json")
    save_json(config, run_root / "resolved_config.json")
    snapshot_sources(run_root, config_path)

    try:
        report = validate_dataset(config, run_root / "dataset_validation.json")
        LOGGER.info(
            "Dataset validation passed: train=%d, val=%d, test=%d",
            report["splits"]["train"]["count"],
            report["splits"]["val"]["count"],
            report["splits"]["test"]["count"],
        )
        if args.check_only:
            save_json({"status": "check_passed"}, run_root / "CHECK_PASSED.json")
            LOGGER.info("Check-only mode completed successfully")
            return 0

        plan = make_plan(config, args.profile, args.smoke_test)
        save_json(
            {
                "profile": args.profile,
                "smoke_test": args.smoke_test,
                "run_count": len(plan),
                "runs": [asdict(spec) | {"run_name": spec.run_name} for spec in plan],
            },
            run_root / "experiment_plan.json",
        )
        LOGGER.info("Experiment plan contains %d runs", len(plan))
        for index, spec in enumerate(plan, start=1):
            LOGGER.info("  %02d. %s", index, spec.run_name)
        if args.dry_run:
            LOGGER.info("Dry-run mode completed; no training was started")
            return 0

        frames = load_split_frames(config)
        runs_root = run_root / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        results = []
        for index, spec in enumerate(plan, start=1):
            LOGGER.info("Pipeline progress %d/%d", index, len(plan))
            current_dir = runs_root / spec.run_name
            result = train_one_run(
                PROJECT_ROOT,
                current_dir,
                config,
                spec,
                frames,
                smoke_test=args.smoke_test,
            )
            analyze_run(current_dir, config)
            results.append(result)

        if not args.smoke_test:
            aggregate_results(run_root, config)
            generate_attention_comparison(PROJECT_ROOT, run_root, config)
            run_controlled_degradation_analysis(PROJECT_ROOT, run_root, config)
            run_lfw_analysis(PROJECT_ROOT, run_root, config)
        else:
            aggregate_results(run_root, config)
        save_json(
            {
                "status": "completed",
                "profile": args.profile,
                "smoke_test": args.smoke_test,
                "completed_runs": len(results),
                "outcome_directory": str(run_root),
            },
            run_root / "PIPELINE_COMPLETED.json",
        )
        LOGGER.info("All experiments and analyses completed successfully")
        return 0
    except Exception as exc:
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_json(failure, run_root / "PIPELINE_FAILED.json")
        LOGGER.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
