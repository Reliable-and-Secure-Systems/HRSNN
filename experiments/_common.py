"""Shared experiment runner used by all entry-point scripts."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config      import load_yaml
from src.utils.device      import get_device
from src.utils.seed        import set_global_seed
from src.utils.logging     import get_logger, setup_file_logging
from src.utils.io          import make_dirs
from src.utils.environment import collect_environment
from src.data.loaders      import build_dataloaders, build_synthetic_loaders
from src.data.registry     import get_dataset, needs_encoder
from src.coding.registry   import get_encoder
from src.models.registry   import get_model
from src.training.trainer  import build_trainer
from src.tracking.tracker  import ExperimentTracker
from src.tracking.activity import measure_activity

logger = get_logger("experiment")


CATEGORY_DIRS: dict[str, str] = {
    "dataset":  "config/datasets",
    "arch":     "config/architectures",
    "coding":   "config/coding_schemes",
    "learning": "config/learning_rules",
}


def resolve_config_path(name_or_path: str, category: str) -> str:
    """Accept either a full path or a short name for a config file."""
    p = Path(name_or_path)
    if p.suffix.lower() == ".yaml" or "/" in name_or_path or "\\" in name_or_path:
        return str(p)
    resolved = Path(CATEGORY_DIRS[category]) / f"{name_or_path}.yaml"
    if not resolved.exists():
        raise FileNotFoundError(
            f"Config '{name_or_path}' not found at {resolved}\n"
            f"Available: {[f.stem for f in Path(CATEGORY_DIRS[category]).glob('*.yaml')]}"
        )
    return str(resolved)


def all_configs(category: str) -> list[str]:
    """Return sorted list of all YAML paths in a config category directory."""
    d = Path(CATEGORY_DIRS[category])
    return sorted(str(f) for f in d.glob("*.yaml"))


def run_experiment(
    ds_path:     str,
    arch_path:   str,
    coding_path: str,
    lr_path:     str,
    *,
    run_name:    str | None = None,
    seed:        int        = 42,
    device_str:  str        = "auto",
    synthetic:   bool       = False,
    results_dir: str        = "results/raw",
) -> dict[str, Any]:
    """Execute one complete SNN experiment end-to-end."""
    ds_cfg     = load_yaml(ds_path)
    arch_cfg   = load_yaml(arch_path)
    coding_cfg = load_yaml(coding_path)
    lr_cfg     = load_yaml(lr_path)

    set_global_seed(seed)
    device = get_device(device_str)

    ds_name     = ds_cfg.get("name", "dataset")
    coding_name = coding_cfg.get("name", "rate")
    lr_name     = lr_cfg.get("name", "surrogate_gradient")
    run_name    = run_name or f"{ds_name}_{coding_name}_{lr_name}_seed{seed}"

    make_dirs(results_dir, "results/figures")
    setup_file_logging(logger, f"{results_dir}/{run_name}.log")
    logger.info("=" * 60)
    logger.info("Run: %s  |  device: %s  |  seed: %d", run_name, device, seed)

    time_steps = coding_cfg.get("time_steps", ds_cfg.get("time_steps", 20))

    if synthetic:
        logger.info("Synthetic data (no download)")
        raw_dim   = ds_cfg.get("input_dim", 784)
        n_classes = ds_cfg.get("n_classes", 10)
        train_loader, val_loader, test_loader = build_synthetic_loaders(
            time_steps = time_steps,
            input_dim  = raw_dim,
            n_classes  = n_classes,
            batch_size = ds_cfg.get("batch_size", 64),
        )
        input_dim = raw_dim

    elif needs_encoder(ds_name):
        encoder = get_encoder(coding_name, **{
            k: v for k, v in coding_cfg.to_dict().items()
            if k not in ("name", "time_steps")
        })
        logger.info("Dataset: %s  Coding: %s  T=%d", ds_name, coding_name, time_steps)

        train_ds = get_dataset(ds_name, train=True,
                               encoder=encoder, time_steps=time_steps,
                               data_dir=ds_cfg.get("data_dir", "data/"))
        test_ds  = get_dataset(ds_name, train=False,
                               encoder=encoder, time_steps=time_steps,
                               data_dir=ds_cfg.get("data_dir", "data/"))

        h, w, c   = train_ds.sensor_size
        raw_dim   = h * w * c
        input_dim = encoder.output_dim(raw_dim)
        n_classes = train_ds.n_classes

        train_loader, val_loader, test_loader = build_dataloaders(
            dataset_train = train_ds,
            dataset_test  = test_ds,
            batch_size    = ds_cfg.get("batch_size", 64),
            val_split     = ds_cfg.get("val_split", 0.1),
            num_workers   = ds_cfg.get("num_workers", 0),
        )

    else:
        logger.info("Loading DVS dataset '%s' ...", ds_name)
        train_ds = get_dataset(ds_name, train=True,
                               data_dir=ds_cfg.get("data_dir", "data/"),
                               time_steps=time_steps)
        test_ds  = get_dataset(ds_name, train=False,
                               data_dir=ds_cfg.get("data_dir", "data/"),
                               time_steps=time_steps)
        h, w, c   = train_ds.sensor_size
        input_dim = h * w * c
        n_classes = train_ds.n_classes

        train_loader, val_loader, test_loader = build_dataloaders(
            dataset_train = train_ds,
            dataset_test  = test_ds,
            batch_size    = ds_cfg.get("batch_size", 64),
            val_split     = ds_cfg.get("val_split", 0.1),
            num_workers   = ds_cfg.get("num_workers", 0),
        )

    logger.info(
        "input_dim=%d  n_classes=%d  train=%d  val=%d  test=%d",
        input_dim, n_classes,
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    arch_name = arch_cfg.get("name", "mlp")
    model     = get_model(arch_name, arch_cfg, input_dim=input_dim, n_classes=n_classes,
                          time_steps=time_steps)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s  (%d params)", arch_name, n_params)

    trainer = build_trainer(model, lr_cfg, device)

    exp_config = {
        "run_name":  run_name,
        "seed":      seed,
        "device":    str(device),
        "synthetic": synthetic,
        "input_dim": input_dim,
        "n_classes": n_classes,
        "dataset":   ds_cfg.to_dict(),
        "coding":    coding_cfg.to_dict(),
        "arch":      arch_cfg.to_dict(),
        "learning":  lr_cfg.to_dict(),
        "environment": collect_environment(),
    }
    tracker = ExperimentTracker(run_name=run_name, config=exp_config,
                                results_dir=results_dir)

    n_epochs = lr_cfg.get("epochs", 10)
    trainer.fit(
        train_loader, val_loader,
        n_epochs     = n_epochs,
        on_epoch_end = tracker.on_epoch_end,
    )

    tracker.load_best_checkpoint(model)
    test_metrics = trainer.evaluate(test_loader)
    activity     = {**measure_activity(model, test_loader, device), "source": "training"}
    tracker.log_final({
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "activity": activity,
    })

    logger.info("Best val acc: %.4f | Test acc: %.4f",
                tracker.best_val_acc(), test_metrics["acc"])

    return {
        "run_name":     run_name,
        "dataset":      ds_name,
        "coding":       coding_name,
        "learning":     lr_name,
        "arch":         arch_name,
        "seed":         seed,
        "test_acc":     test_metrics["acc"],
        "test_loss":    test_metrics["loss"],
        "best_val_acc": tracker.best_val_acc(),
        "n_params":     n_params,
        "input_dim":    input_dim,
        "n_classes":    n_classes,
    }
