"""Integer weight-memory fault injectors: a known fault produces the expected weight, sites are uniform over the memory, and the transient and permanent injectors share one substrate."""
from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from fault_injection.transient.bit_flip import (
    INT8BitFlipInjector, _int_weight_fault, _sample_memory_sites,
)
from fault_injection.permanent.stuck_at_weights import IntStuckAtWeights
from hardware import analog_imc, analog_neuromorphic, digital_imc, digital_neuromorphic


def _sites(*pairs):
    elem = torch.tensor([e for e, _ in pairs], dtype=torch.int64)
    bit  = torch.tensor([b for _, b in pairs], dtype=torch.int64)
    return elem, bit


class IntWeightFaultTest(unittest.TestCase):
    """8-bit weights [127, 1, -1, 0] * s quantize to exactly [127, 1, -1, 0]."""

    def setUp(self):
        self.w   = torch.tensor([127.0, 1.0, -1.0, 0.0]) * 0.01
        self.lsb = 0.01

    def _apply(self, fault_type, *pairs):
        elem, bit = _sites(*pairs)
        out = _int_weight_fault(self.w, elem, bit, torch.Generator().manual_seed(0),
                                bits=8, fault_type=fault_type)
        return torch.round(out / self.lsb).to(torch.int64).tolist()

    def test_no_sites_is_pure_quantization(self):
        self.assertEqual(self._apply("flip"), [127, 1, -1, 0])

    def test_flip_toggles_the_bit(self):
        self.assertEqual(self._apply("flip", (1, 1), (3, 7)), [127, 3, -1, -128])

    def test_stuck_at_zero_clears_the_bit(self):
        self.assertEqual(self._apply("sa0", (0, 6), (2, 7)), [63, 1, 127, 0])

    def test_stuck_at_one_sets_the_bit(self):
        self.assertEqual(self._apply("sa1", (1, 7), (3, 0)), [127, -127, -1, 1])

    def test_stuck_at_matching_value_is_harmless(self):
        self.assertEqual(self._apply("sa1", (1, 0)), [127, 1, -1, 0])
        self.assertEqual(self._apply("sa0", (3, 5)), [127, 1, -1, 0])

    def test_unknown_fault_type_rejected(self):
        with self.assertRaises(ValueError):
            self._apply("toggle")


class SiteSamplingTest(unittest.TestCase):

    def test_sites_distinct_and_counted(self):
        sites = _sample_memory_sites([1000, 50, 7], bits=8, n_sites=400,
                                     rng=torch.Generator().manual_seed(1))
        self.assertEqual(sum(e.numel() for e, _ in sites), 400)
        for (elem, bit), size in zip(sites, [1000, 50, 7]):
            codes = (elem * 8 + bit).tolist()
            self.assertEqual(len(codes), len(set(codes)))
            self.assertTrue(all(0 <= e < size for e in elem.tolist()))
            self.assertTrue(all(0 <= b < 8 for b in bit.tolist()))

    def test_allocation_proportional_to_layer_size(self):
        sites = _sample_memory_sites([99_000, 1_000], bits=8, n_sites=5_000,
                                     rng=torch.Generator().manual_seed(2))
        small = sites[1][0].numel()
        self.assertLess(small, 150)
        self.assertGreater(small, 10)

    def test_zero_sites(self):
        sites = _sample_memory_sites([10, 20], bits=4, n_sites=0,
                                     rng=torch.Generator())
        self.assertEqual([e.numel() for e, _ in sites], [0, 0])


class InjectorTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.model = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 10))
        self.orig  = [p.detach().clone() for p in self.model.parameters()]

    def _weights_after(self, injector):
        injector.inject(self.model)
        w = [m.weight.detach().clone() for m in self.model]
        injector.restore(self.model)
        return w

    def test_restore_is_exact(self):
        for inj in (INT8BitFlipInjector(n_flips=50, seed=3, bits=6, weight_noise_std=0.02),
                    IntStuckAtWeights(n_faults=50, seed=3, bits=4, weight_noise_std=0.05)):
            self._weights_after(inj)
            for a, b in zip(self.orig, self.model.parameters()):
                self.assertTrue(torch.equal(a, b))

    def test_zero_faults_give_the_same_substrate(self):
        for bits, noise in [(8, 0.0), (6, 0.02), (4, 0.05)]:
            flip  = self._weights_after(INT8BitFlipInjector(0, seed=7, bits=bits, weight_noise_std=noise))
            stuck = self._weights_after(IntStuckAtWeights(0, seed=7, bits=bits, weight_noise_std=noise))
            for a, b in zip(flip, stuck):
                self.assertTrue(torch.equal(a, b))

    def test_seed_reproducible(self):
        a = self._weights_after(IntStuckAtWeights(n_faults=30, seed=11))
        b = self._weights_after(IntStuckAtWeights(n_faults=30, seed=11))
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))

    def test_stuck_at_rejects_flip(self):
        with self.assertRaises(ValueError):
            IntStuckAtWeights(n_faults=1, fault_type="flip")


class SubstrateTest(unittest.TestCase):
    """Every fault class of a substrate uses that substrate's memory model."""

    def test_fault_classes_share_bits_and_noise(self):
        substrates = [
            ([digital_imc.DigitalIMCTransientFault,
              digital_imc.DigitalIMCMembraneFault,
              digital_imc.DigitalIMCBlockFault], 4, 0.0),
            ([digital_neuromorphic.DigitalNeuromorphicTransientFault,
              digital_neuromorphic.DigitalNeuromorphicMembraneFault,
              digital_neuromorphic.DigitalNeuromorphicBlockFault], 8, 0.0),
            ([analog_imc.AnalogIMCStuckOnFault, analog_imc.AnalogIMCStuckOffFault], 6, 0.02),
            ([analog_neuromorphic.AnalogNeuromorphicStuckOnFault,
              analog_neuromorphic.AnalogNeuromorphicStuckOffFault], 4, 0.05),
        ]
        for classes, bits, noise in substrates:
            for cls in classes:
                inj = cls()
                self.assertEqual((inj.bits, inj.weight_noise_std), (bits, noise), cls.__name__)

    def test_fault_types(self):
        self.assertEqual(digital_imc.DigitalIMCTransientFault().fault_type, "flip")
        self.assertEqual(digital_imc.DigitalIMCMembraneFault().fault_type, "membrane_flip")
        self.assertEqual(digital_imc.DigitalIMCBlockFault().fault_type, "block_stuck")
        self.assertEqual(analog_imc.AnalogIMCStuckOnFault().fault_type, "stuck_on")
        self.assertEqual(analog_imc.AnalogIMCStuckOffFault().fault_type, "stuck_off")


if __name__ == "__main__":
    unittest.main()
