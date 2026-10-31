from __future__ import annotations
import torch
import torch.nn as nn
from src.learning.base import BaseLearner


class SurrogateGradientLearner(BaseLearner):
    """BPTT with surrogate gradients (AdamW, AMP, cosine/step/plateau scheduler)."""

    def __init__(
        self,
        lr: float             = 1e-3,
        weight_decay: float   = 0.0,
        grad_clip: float      = 5.0,
        scheduler: str | None = None,
        max_epochs: int       = 50,
        lr_step_size: int     = 10,
        lr_gamma: float       = 0.5,
    ):
        self.lr             = lr
        self.weight_decay   = weight_decay
        self.grad_clip      = grad_clip
        self.scheduler_type = scheduler
        self.max_epochs     = max_epochs
        self.lr_step_size   = lr_step_size
        self.lr_gamma       = lr_gamma

        self.optimizer  = None
        self.scheduler  = None
        self.criterion  = nn.CrossEntropyLoss()
        self._scaler    = None
        self._use_amp   = False
        self.device     = None

    def setup(self, model: nn.Module, cfg, device: torch.device):
        if cfg is not None:
            self.lr             = cfg.get("lr",           self.lr)
            self.weight_decay   = cfg.get("weight_decay", self.weight_decay)
            self.grad_clip      = cfg.get("grad_clip",    self.grad_clip)
            self.scheduler_type = cfg.get("scheduler",    self.scheduler_type)
            self.max_epochs     = cfg.get("epochs",       self.max_epochs)

        self.device   = device
        self._use_amp = (device.type == "cuda")
        self._scaler  = torch.amp.GradScaler("cuda", enabled=self._use_amp)

        opt_name = (cfg.get("optimizer", "adamw") if cfg else "adamw").lower()
        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif opt_name == "sgd":
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9
            )
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        if self.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.max_epochs
            )
        elif self.scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma
            )
        elif self.scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=self.lr_gamma, patience=5
            )
        else:
            self.scheduler = None

    def step(
        self,
        spikes: torch.Tensor,
        labels: torch.Tensor,
        model: nn.Module,
    ) -> tuple[float, torch.Tensor]:
        """One surrogate-gradient step on a batch; returns (loss, logits)."""
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
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_acc)
        else:
            self.scheduler.step()

    @property
    def current_lr(self) -> float:
        if self.optimizer is None:
            return self.lr
        return self.optimizer.param_groups[0]["lr"]
