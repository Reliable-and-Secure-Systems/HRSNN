"""Training and evaluation engine for SNN models — pure-function core."""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.learning.base     import BaseLearner
from src.learning.registry import get_learner
from src.utils.logging     import get_logger

logger = get_logger(__name__)


def train_one_epoch(
    model:     nn.Module,
    learner:   BaseLearner,
    loader:    DataLoader,
    device:    torch.device,
    callbacks: Sequence = (),
) -> Dict[str, float]:
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch_idx, (spikes, labels) in enumerate(loader):
        spikes = spikes.to(device).permute(1, 0, 2).contiguous()
        labels = labels.to(device).long()

        loss_val, logits = learner.step(spikes, labels, model)

        B           = labels.size(0)
        total_loss += loss_val * B
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += B

        for cb in callbacks:
            cb.on_batch_end(batch_idx, {"loss": loss_val})

    n = max(total, 1)
    return {"loss": total_loss / n, "acc": correct / n}


def evaluate(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
    loss_fn: Callable | None = None,
) -> Dict[str, float]:
    """Evaluation pass (no grad)."""
    if loss_fn is None:
        loss_fn = lambda spk, lbl: F.cross_entropy(spk.sum(dim=0), lbl)

    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for spikes, labels in loader:
            spikes = spikes.to(device).permute(1, 0, 2).contiguous()
            labels = labels.to(device).long()

            spk_rec, _ = model(spikes)
            logits     = spk_rec.sum(dim=0)
            loss_val   = loss_fn(spk_rec, labels).item()

            B           = labels.size(0)
            total_loss += loss_val * B
            correct    += (logits.argmax(dim=1) == labels).sum().item()
            total      += B

    n = max(total, 1)
    return {"loss": total_loss / n, "acc": correct / n}


def fit(
    model:        nn.Module,
    learner:      BaseLearner,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    n_epochs:     int,
    device:       torch.device,
    callbacks:    Sequence = (),
    on_epoch_end: Callable | None = None,
) -> List[Dict]:
    """Full training loop built from train_one_epoch + evaluate."""
    history: List[Dict] = []

    for epoch in range(1, n_epochs + 1):
        t0            = time.time()
        train_metrics = train_one_epoch(model, learner, train_loader, device, callbacks)
        val_metrics   = evaluate(model, val_loader, device)

        learner.on_epoch_end(epoch, val_metrics["acc"])

        row = {
            "epoch":      epoch,
            "train_loss": train_metrics["loss"],
            "train_acc":  train_metrics["acc"],
            "val_loss":   val_metrics["loss"],
            "val_acc":    val_metrics["acc"],
            "lr":         learner.current_lr,
            "time_s":     round(time.time() - t0, 2),
        }
        history.append(row)

        logger.info(
            "Epoch %3d/%d | train_loss=%.4f  train_acc=%.4f | "
            "val_loss=%.4f  val_acc=%.4f | lr=%.2e | %.1fs",
            epoch, n_epochs,
            row["train_loss"], row["train_acc"],
            row["val_loss"],   row["val_acc"],
            row["lr"],         row["time_s"],
        )

        if on_epoch_end is not None:
            on_epoch_end(row, model)

        for cb in callbacks:
            if not cb.on_epoch_end(epoch, row, model):
                logger.info("Early stopping triggered at epoch %d", epoch)
                return history

    return history


class Trainer:
    """Convenience wrapper that holds (model, learner, device, callbacks) and exposes fit() / evaluate() as methods."""

    def __init__(
        self,
        model:     nn.Module,
        learner:   BaseLearner,
        device:    torch.device,
        callbacks: Sequence = (),
    ):
        self.model     = model.to(device)
        self.learner   = learner
        self.device    = device
        self.callbacks = list(callbacks)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        n_epochs:     int,
        on_epoch_end: Callable | None = None,
    ) -> List[Dict]:
        return fit(
            self.model, self.learner,
            train_loader, val_loader,
            n_epochs, self.device,
            self.callbacks, on_epoch_end,
        )

    def evaluate(self, loader: DataLoader) -> Dict:
        return evaluate(self.model, loader, self.device)


def build_trainer(
    model:     nn.Module,
    cfg,
    device:    torch.device,
    callbacks: Sequence = (),
) -> Trainer:
    """Build a Trainer from a learning-rule ConfigDict."""
    name    = cfg.get("name", "surrogate_gradient")
    learner = get_learner(name)
    learner.setup(model, cfg, device)
    return Trainer(model=model, learner=learner, device=device, callbacks=callbacks)
