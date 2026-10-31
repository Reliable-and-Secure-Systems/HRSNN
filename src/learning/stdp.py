from __future__ import annotations
import torch
import torch.nn as nn
from src.learning.base import BaseLearner


class STDPLearner(BaseLearner):
    """Unsupervised STDP on the first hidden layer, then supervised backprop training of the rest."""

    def __init__(
        self,
        a_plus: float      = 0.01,
        a_minus: float     = 0.0105,
        lr: float          = 1e-3,
        stdp_lr: float     = 3e-2,
        trace_decay: float = 0.8,
        k_winners: float   = 0.05,
        weight_clip: float = 1.0,
        stdp_epochs: int   = 1,
    ):
        self.a_plus      = a_plus
        self.a_minus     = a_minus
        self.lr          = lr
        self.stdp_lr     = stdp_lr
        self.trace_decay = trace_decay
        self.k_winners   = k_winners
        self.weight_clip = weight_clip
        self.stdp_epochs = stdp_epochs
        self._epochs_done = 0
        self._layer: nn.Linear | None = None
        self._block: nn.Module | None = None
        self._row_norm: torch.Tensor | None = None
        self._readout_opt = None
        self._post: list[torch.Tensor] = []
        self.criterion = nn.CrossEntropyLoss()

    def setup(self, model: nn.Module, cfg, device: torch.device):
        if cfg is not None:
            for key in ("a_plus", "a_minus", "lr", "stdp_lr", "trace_decay",
                        "k_winners", "weight_clip", "stdp_epochs"):
                setattr(self, key, cfg.get(key, getattr(self, key)))
        self.device = device

        blocks = getattr(model, "blocks", None)
        if not blocks:
            raise ValueError("STDPLearner needs a model with .blocks (SNNMLP)")
        self._block = blocks[0]
        self._layer = self._block.fc
        self._row_norm = self._layer.weight.detach().norm(dim=1, keepdim=True).clone()

        def record(module, inputs, output):
            if module.training and self.in_stdp_phase:
                self._post.append(output[0].detach())
        self._block.register_forward_hook(record)

        self._layer.weight.requires_grad_(False)
        readout_params = [p for p in model.parameters() if p.requires_grad]
        self._readout_opt = torch.optim.Adam(readout_params, lr=self.lr)

    @property
    def in_stdp_phase(self) -> bool:
        """True during the unsupervised epochs (phase 1)."""
        return self._epochs_done < self.stdp_epochs

    @torch.no_grad()
    def stdp_update(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """Apply one STDP step to the first layer and return ΔW (before scaling)."""
        T, B, n_out = post.shape
        k = max(1, int(round(self.k_winners * n_out)))
        counts = post.sum(dim=0)
        winners = torch.zeros_like(counts)
        top = counts.topk(k, dim=1).indices
        winners.scatter_(1, top, 1.0)
        winners = winners * (counts > 0)
        post = post * winners.unsqueeze(0)

        pre_tr = torch.zeros_like(pre[0])
        post_tr = torch.zeros_like(post[0])
        dw = torch.zeros_like(self._layer.weight)
        for t in range(T):
            pre_tr = self.trace_decay * pre_tr + pre[t]
            dw += self.a_plus * post[t].t() @ pre_tr
            dw -= self.a_minus * post_tr.t() @ pre[t]
            post_tr = self.trace_decay * post_tr + post[t]
        dw /= B

        w = self._layer.weight
        if self._row_norm.device != w.device:
            self._row_norm = self._row_norm.to(w.device)
        w.add_(self.stdp_lr * dw)
        w.mul_(self._row_norm / w.norm(dim=1, keepdim=True).clamp_min(1e-12))
        w.clamp_(-self.weight_clip, self.weight_clip)
        return dw

    def step(
        self,
        spikes: torch.Tensor,
        labels: torch.Tensor,
        model: nn.Module,
    ) -> tuple[float, torch.Tensor]:
        """STDP update of the first layer in phase 1, backprop of the other layers in phase 2; returns (loss, logits)."""
        model.train()
        if self.in_stdp_phase:
            self._post = []
            with torch.no_grad():
                spk_rec, _ = model(spikes)
                logits = spk_rec.mean(dim=0)
                loss = self.criterion(logits, labels)
            post = torch.stack(self._post)
            self._post = []
            self.stdp_update(spikes.float(), post.float())
            return loss.item(), logits.detach()

        model.zero_grad(set_to_none=True)
        spk_rec, _ = model(spikes)
        logits = spk_rec.mean(dim=0)
        loss = self.criterion(logits, labels)
        loss.backward()
        self._readout_opt.step()
        return loss.item(), logits.detach()

    def on_epoch_end(self, epoch: int, val_acc: float):
        self._epochs_done += 1

    @property
    def current_lr(self) -> float:
        """Adam learning rate of the backprop-trained parameters (logged per epoch)."""
        return self.lr


class RewardModulatedSTDP(BaseLearner):
    """R-STDP: STDP gated by a reward signal derived from validation accuracy."""

    def __init__(
        self,
        a_plus: float      = 0.01,
        a_minus: float     = 0.005,
        lr: float          = 1e-3,
        weight_clip: float = 1.0,
        baseline: float    = 0.5,
    ):
        self.a_plus      = a_plus
        self.a_minus     = a_minus
        self.lr          = lr
        self.weight_clip = weight_clip
        self.baseline    = baseline
        self._reward: float        = 0.0
        self._layer: nn.Linear | None = None

    def setup(self, model: nn.Module, cfg, device: torch.device):
        if cfg is not None:
            self.a_plus      = cfg.get("a_plus",      self.a_plus)
            self.a_minus     = cfg.get("a_minus",      self.a_minus)
            self.lr          = cfg.get("lr",           self.lr)
            self.weight_clip = cfg.get("weight_clip",  self.weight_clip)
            self.baseline    = cfg.get("baseline",     self.baseline)

        self.device = device

        for m in model.modules():
            if isinstance(m, nn.Linear):
                self._layer = m
                break
        if self._layer is None:
            raise ValueError("RewardModulatedSTDP: no Linear layer found in model")

    def step(
        self,
        spikes: torch.Tensor,
        labels: torch.Tensor,
        model: nn.Module,
    ) -> tuple[float, torch.Tensor]:
        """Reward-gated STDP update of the layer, then a no-grad forward pass; returns (0.0, logits)."""
        pre  = spikes.float().mean(dim=0)
        post = torch.relu(self._layer(pre))
        dW_plus  = self.a_plus  * post.t() @ pre
        dW_minus = self.a_minus * pre.t()  @ post
        dW       = dW_plus - dW_minus.t()

        with torch.no_grad():
            self._layer.weight.data.add_(self.lr * self._reward * dW)
            self._layer.weight.data.clamp_(-self.weight_clip, self.weight_clip)

        with torch.no_grad():
            spk_rec, _ = model(spikes)
            logits = spk_rec.sum(dim=0)

        return 0.0, logits.detach()

    def on_epoch_end(self, epoch: int, val_acc: float):
        """Update reward as deviation from exponential moving baseline."""
        self._reward  = val_acc - self.baseline
        self.baseline = 0.9 * self.baseline + 0.1 * val_acc
