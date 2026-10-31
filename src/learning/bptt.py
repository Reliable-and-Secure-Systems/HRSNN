from __future__ import annotations
import torch
import torch.nn as nn
from src.learning.base import BaseLearner


class BPTTLearner(BaseLearner):
    """Full back-propagation through time (Adam, AMP, cosine scheduler)."""

    def __init__(
        self,
        lr: float           = 1e-3,
        weight_decay: float = 0.0,
        grad_clip: float    = 5.0,
        max_epochs: int     = 50,
    ):
        self.lr           = lr
        self.weight_decay = weight_decay
        self.grad_clip    = grad_clip
        self.max_epochs   = max_epochs

        self.optimizer  = None
        self.scheduler  = None
        self.criterion  = nn.CrossEntropyLoss()
        self._scaler    = None
        self._use_amp   = False
        self.device     = None

    def setup(self, model: nn.Module, cfg, device: torch.device):
        if cfg is not None:
            self.lr           = cfg.get("lr",           self.lr)
            self.weight_decay = cfg.get("weight_decay", self.weight_decay)
            self.grad_clip    = cfg.get("grad_clip",    self.grad_clip)
            self.max_epochs   = cfg.get("epochs",       self.max_epochs)

        self.device   = device
        self._use_amp = (device.type == "cuda")
        self._scaler  = torch.amp.GradScaler("cuda", enabled=self._use_amp)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs
        )

    def step(
        self,
        spikes: torch.Tensor,
        labels: torch.Tensor,
        model: nn.Module,
    ) -> tuple[float, torch.Tensor]:
        """One backprop-through-time step on a batch; returns (loss, logits)."""
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=self.device.type, enabled=self._use_amp):
            spk_rec, _ = model(spikes)
            logits = spk_rec.sum(dim=0)
            loss   = self.criterion(logits, labels)

        self._scaler.scale(loss).backward()
        if self.grad_clip > 0:
            self._scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        self._scaler.step(self.optimizer)
        self._scaler.update()

        return loss.item(), logits.detach()

    def on_epoch_end(self, epoch: int, val_acc: float):
        if self.scheduler:
            self.scheduler.step()

    @property
    def current_lr(self) -> float:
        if self.optimizer is None:
            return self.lr
        return self.optimizer.param_groups[0]["lr"]
