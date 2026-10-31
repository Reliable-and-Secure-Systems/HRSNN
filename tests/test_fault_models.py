"""Memory-block permanent faults, membrane-register upsets and stuck memristor cells."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from analysis.statistical_tests import _signed_rank_biserial
from experiments.run_fault_evaluation import (
    FAULT_MODEL_VERSION, FAULT_SWEEPS, _pending_sweeps,
)
from fault_injection.permanent.memory_block import MemoryBlockStuckAt
from fault_injection.permanent.memristor_stuck import MemristorStuckAt, stuck_cells
from fault_injection.transient.bit_flip import INT8BitFlipInjector, IntWeightFaultInjector
from fault_injection.transient.membrane_seu import (
    MembraneSEU, flip_membrane_bits, register_lsb,
)
from src.models.mlp import SNNMLP


def _weights(model):
    return [m.weight.detach().clone() for m in model.modules() if isinstance(m, nn.Linear)]


def _after(injector, model):
    injector.inject(model)
    w = _weights(model)
    injector.restore(model)
    return w


class MemoryBlockTest(unittest.TestCase):
    """W [out=5, in=6], 4-bit, tiles of 4 synapses."""

    def setUp(self):
        self.layer = nn.Linear(6, 5, bias=False)
        self.layers = [("fc", self.layer)]

    def _plan(self, n, row_share, seed=0):
        inj = MemoryBlockStuckAt(n_faulty_bits=n, bits=4, tile=4, row_share=row_share)
        return inj, inj._plan(self.layers, torch.Generator().manual_seed(seed))[0]

    def test_row_block_is_one_input_one_tile_all_bits(self):
        inj, p = self._plan(1, row_share=1.0)
        self.assertEqual(len(inj.blocks), 1)
        outs, ins = p["elem"] // 6, p["elem"] % 6
        self.assertEqual(len(set(ins.tolist())), 1)
        self.assertLessEqual(len(set(outs.tolist())), 4)
        self.assertEqual(max(outs.tolist()) - min(outs.tolist()) + 1, len(set(outs.tolist())))
        for o in set(outs.tolist()):
            self.assertEqual(sorted(p["bit"][outs == o].tolist()), [0, 1, 2, 3])
        self.assertEqual(len(set(p["to_one"].tolist())), 1)

    def test_column_block_is_one_output_one_bit_one_tile(self):
        inj, p = self._plan(1, row_share=0.0)
        outs, ins = p["elem"] // 6, p["elem"] % 6
        self.assertEqual(len(set(outs.tolist())), 1)
        self.assertEqual(len(set(p["bit"].tolist())), 1)
        self.assertLessEqual(len(set(ins.tolist())), 4)
        self.assertEqual(len(set(p["to_one"].tolist())), 1)

    def test_target_reached_with_distinct_sites(self):
        _, p = self._plan(40, row_share=0.6, seed=3)
        self.assertGreaterEqual(p["elem"].numel(), 40 - 4)
        codes = (p["elem"] * 4 + p["bit"]).tolist()
        self.assertEqual(len(codes), len(set(codes)))

    def test_zero_target_is_fault_free(self):
        inj, p = self._plan(0, row_share=0.6)
        self.assertEqual(p["elem"].numel(), 0)
        self.assertEqual(inj.blocks, [])

    def test_row_stuck_at_zero_disconnects_its_synapses(self):
        torch.manual_seed(0)
        inj = MemoryBlockStuckAt(n_faulty_bits=1, bits=4, tile=4, row_share=1.0, seed=0)
        model = nn.Sequential(self.layer)
        for seed in range(50):
            inj.seed = seed
            plan = inj._plan([("0", self.layer)], torch.Generator().manual_seed(seed))[0]
            if not plan["to_one"][0]:
                break
        w = _after(inj, model)[0].flatten()
        self.assertTrue(torch.all(w[plan["elem"].unique()] == 0))


class MembraneSEUTest(unittest.TestCase):

    def test_register_spans_256_thresholds(self):
        lsb = register_lsb(1.0)
        self.assertAlmostEqual(lsb, 2.0 ** -15)
        mem = torch.tensor([[0.5, 0.25]])
        out = flip_membrane_bits(mem, torch.tensor([0]), torch.tensor([1 << 22]), lsb, 24)
        self.assertAlmostEqual(out[0, 0].item(), 0.5 + 128.0, places=4)
        self.assertEqual(out[0, 1].item(), 0.25)
        out = flip_membrane_bits(mem, torch.tensor([0]), torch.tensor([1 << 23]), lsb, 24)
        self.assertAlmostEqual(out[0, 0].item(), 0.5 - 256.0, places=4)

    def test_operating_range_is_not_clamped(self):
        lsb = register_lsb(1.0)
        mem = torch.tensor([[-170.0]])
        out = flip_membrane_bits(mem, torch.tensor([0]), torch.tensor([1 << 0]), lsb, 24)
        self.assertAlmostEqual(out[0, 0].item(), -170.0, delta=2 * lsb)

    def test_no_upset_after_the_last_step(self):
        seu = MembraneSEU(n_upsets=2000, seed=3)
        seu._rng = torch.Generator().manual_seed(3)
        upsets = seu._draw([16, 8], time_steps=10)
        steps = {t for per_block in upsets.values() for t in per_block}
        self.assertTrue(steps)
        self.assertLessEqual(max(steps), 8)

    def setUp(self):
        torch.manual_seed(0)
        self.model = SNNMLP(input_dim=20, hidden_dims=[16, 8], n_classes=3).eval()
        self.x = (torch.rand(10, 4, 20) < 0.3).float()

    def _out(self, injector=None):
        if injector is not None:
            injector.inject(self.model)
        with torch.no_grad():
            out, _ = self.model(self.x)
        if injector is not None:
            injector.restore(self.model)
        return out

    def test_zero_upsets_equals_substrate(self):
        sub = IntWeightFaultInjector(n_sites=0, bits=8)
        seu = MembraneSEU(n_upsets=0, substrate=IntWeightFaultInjector(n_sites=0, bits=8))
        self.assertTrue(torch.equal(self._out(sub), self._out(seu)))

    def test_upsets_change_output_and_restore(self):
        clean = self._out()
        seu = MembraneSEU(n_upsets=2000, seed=1)
        faulty = self._out(seu)
        self.assertFalse(torch.equal(clean, faulty))
        self.assertTrue(torch.equal(clean, self._out()))
        stats = seu.fault_stats()
        self.assertEqual(stats["placed"], 2000)
        self.assertEqual(stats["clamped_fraction"], 0.0)

    def test_fault_sites(self):
        self.assertEqual(MembraneSEU().fault_sites(self.model, time_steps=10),
                         (16 + 8) * 24 * 9)


class MemristorTest(unittest.TestCase):

    W = torch.tensor([0.5, -0.5, 1.0, 0.0])

    def _stuck(self, elem, cell, state):
        return stuck_cells(self.W, torch.tensor([elem]), torch.tensor([cell]), state).tolist()

    def test_stuck_on(self):
        self.assertEqual(self._stuck(0, 0, "on")[0], 1.0)
        self.assertEqual(self._stuck(1, 0, "on")[1], 0.5)
        self.assertEqual(self._stuck(1, 1, "on")[1], -1.0)

    def test_stuck_off(self):
        self.assertEqual(self._stuck(0, 0, "off")[0], 0.0)
        self.assertEqual(self._stuck(1, 0, "off")[1], -0.5)
        self.assertEqual(self._stuck(1, 1, "off")[1], 0.0)

    def test_zero_cells_equals_analog_substrate_including_noise(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(30, 20), nn.Linear(20, 5))
        sub = _after(IntWeightFaultInjector(n_sites=0, seed=3, bits=4, weight_noise_std=0.05), model)
        mem = _after(MemristorStuckAt(n_cells=0, seed=3, bits=4, weight_noise_std=0.05), model)
        for a, b in zip(sub, mem):
            self.assertTrue(torch.equal(a, b))

    def test_fault_sites_are_two_cells_per_weight(self):
        model = nn.Sequential(nn.Linear(30, 20), nn.Linear(20, 5))
        self.assertEqual(MemristorStuckAt().fault_sites(model), 2 * (30 * 20 + 20 * 5))


class FaultCountTest(unittest.TestCase):
    """Placed vs changed: what the evaluator records next to each rate."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = nn.Sequential(nn.Linear(300, 200), nn.Linear(200, 10))

    def _stats(self, inj):
        inj.inject(self.model)
        inj.restore(self.model)
        return inj.fault_stats()

    def test_flips_change_every_placed_bit(self):
        st = self._stats(INT8BitFlipInjector(n_flips=500, seed=1, bits=8))
        self.assertEqual(st["placed"], 500)
        self.assertEqual(st["changed"], 500)

    def test_blocks_overshoot_and_half_change(self):
        st = self._stats(MemoryBlockStuckAt(n_faulty_bits=500, seed=1, bits=8))
        self.assertGreaterEqual(st["placed"], 500)
        self.assertLess(st["changed"], st["placed"])

    def test_stuck_cell_on_idle_half_changes_nothing(self):
        stats = {}
        w = torch.tensor([0.5, -0.5])
        out = stuck_cells(w, torch.tensor([0]), torch.tensor([1]), "off", stats=stats)
        self.assertTrue(torch.equal(out, w))
        self.assertEqual((stats["placed"], stats["changed"]), (1, 0))


class PendingSweepTest(unittest.TestCase):
    """A quick check with fewer trials must not count as the full evaluation."""

    def test_settings_must_match(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fault_results.json"
            path.write_text(json.dumps({"fault_sweeps": {"digital_imc": {
                "fault_model_version": FAULT_MODEL_VERSION,
                "n_trials": 2, "max_batches": 10}}}))
            self.assertEqual(_pending_sweeps(path, ["digital_imc"], False, 10, None),
                             ["digital_imc"])
            self.assertEqual(_pending_sweeps(path, ["digital_imc"], False, 2, 10), [])


class EffectSizeTest(unittest.TestCase):

    def test_sign_follows_direction(self):
        self.assertEqual(_signed_rank_biserial([2, 3, 4, 5, 6], [1, 2, 3, 4, 5]), 1.0)
        self.assertEqual(_signed_rank_biserial([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]), -1.0)


class SweepBaselineTest(unittest.TestCase):
    """run_fault_evaluation.py computes one baseline per substrate and reuses it."""

    def test_sweeps_of_a_substrate_share_the_zero_fault_baseline(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(40, 30), nn.Linear(30, 10))
        by_hw = {}
        for name, sweep in FAULT_SWEEPS.items():
            inj = sweep["build"](0, None)
            inj.seed = 7
            by_hw.setdefault(sweep["hardware"], []).append((name, _after(inj, model)))
        for hw, entries in by_hw.items():
            ref_name, ref = entries[0]
            for name, w in entries[1:]:
                for a, b in zip(ref, w):
                    self.assertTrue(torch.equal(a, b), f"{hw}: {name} vs {ref_name}")


if __name__ == "__main__":
    unittest.main()
