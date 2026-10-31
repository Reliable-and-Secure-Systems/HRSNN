from __future__ import annotations

import torch
import torch.nn as nn

from src.learning.base import BaseLearner


class EPropLearner(BaseLearner):
    """Online eligibility propagation (e-prop) for feedforward SNNs."""

    def __init__(
        self,
        lr:           float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip:    float = 5.0,
        max_epochs:   int   = 50,
    ):
        self.lr           = lr
        self.weight_decay = weight_decay
        self.grad_clip    = grad_clip
        self.max_epochs   = max_epochs

        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler                               = None
        self.criterion  = nn.CrossEntropyLoss()
        self._scaler    = None
        self._use_amp   = False
        self.device: torch.device | None             = None


    def setup(self, model: nn.Module, cfg, device: torch.device) -> None:
        _check_model(model)

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
        model:  nn.Module,
    ) -> tuple[float, torch.Tensor]:
        """One e-prop step on a batch with a per-time-step loss; returns (loss, logits)."""
        T = spikes.shape[0]
        model.train()
        self.optimizer.zero_grad(set_to_none=True)

        model.reset_states()

        total_loss    = 0.0
        logit_list: list[torch.Tensor] = []

        for t in range(T):
            cur = spikes[t]

            with torch.amp.autocast(
                device_type=self.device.type, enabled=self._use_amp
            ):
                for block in model.blocks:
                    cur, _ = block(cur)

                logits_t = model.output_fc(cur)

                loss_t = self.criterion(logits_t, labels) / T

            self._scaler.scale(loss_t).backward()
            total_loss += loss_t.item()

            for block in model.blocks:
                if block.mem is not None:
                    block.mem = block.mem.detach()

            logit_list.append(logits_t.detach())

        if self.grad_clip > 0:
            self._scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        self._scaler.step(self.optimizer)
        self._scaler.update()

        logits = torch.stack(logit_list).mean(dim=0)
        return total_loss, logits.detach()

    def on_epoch_end(self, epoch: int, val_acc: float) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    @property
    def current_lr(self) -> float:
        if self.optimizer is None:
            return self.lr
        return self.optimizer.param_groups[0]["lr"]


def _check_model(model: nn.Module) -> None:
    missing = [
        a for a in ("blocks", "output_fc", "reset_states")
        if not hasattr(model, a)
    ]
    if missing:
        raise AttributeError(
            f"EPropLearner requires model attributes "
            f"['blocks', 'output_fc', 'reset_states']. "
            f"Missing: {missing}. Use SNNMLP."
        )
