from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config, save_config
from src.dataset import HER2PatchDataset, read_manifest, samples_for_split, scan_directory_dataset, split_train_val, validate_samples
from src.logging_utils import setup_logger
from src.reproducibility import environment_metadata, seed_everything
from src.smoke_tests import run_smoke_tests
from src.train import train_hybrid
from src.transforms import build_transforms
from src.utils import next_experiment_dir, write_json


def load_samples(config):
    paths = config["paths"]
    if paths.get("train_manifest") and paths.get("val_manifest") and paths.get("test_manifest"):
        samples = read_manifest(paths["train_manifest"]) + read_manifest(paths["val_manifest"]) + read_manifest(paths["test_manifest"])
    else:
        samples = scan_directory_dataset(paths["data_dir"], config)
        samples = split_train_val(samples, float(config["dataset"]["val_fraction"]), int(config["seed"]))
    validate_samples(samples, config)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hybrid.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-smoke-tests", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_dir:
        config["paths"]["data_dir"] = args.data_dir
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir

    seed_everything(int(config["seed"]))
    exp_dir = next_experiment_dir(config["paths"]["output_dir"], config["experiment_prefix"], config["experiment_name"])
    save_config(config, exp_dir / "config.yaml")
    write_json(environment_metadata(), exp_dir / "environment.json")
    logger = setup_logger(exp_dir / "training.log")
    logger.info("Experiment directory: %s", exp_dir)

    samples = load_samples(config)
    class_to_idx = config["classes"]
    train_dataset = HER2PatchDataset(samples_for_split(samples, "train"), class_to_idx, build_transforms(config, "train"))
    val_dataset = HER2PatchDataset(samples_for_split(samples, "val"), class_to_idx, build_transforms(config, "val"))
    test_dataset = HER2PatchDataset(samples_for_split(samples, "test"), class_to_idx, build_transforms(config, "test"))

    if not args.skip_smoke_tests:
        run_smoke_tests(config, train_dataset, val_dataset, Path(exp_dir))
    metrics = train_hybrid(config, train_dataset, val_dataset, test_dataset, exp_dir, logger)
    logger.info("Final test metrics: %s", metrics)


if __name__ == "__main__":
    main()

