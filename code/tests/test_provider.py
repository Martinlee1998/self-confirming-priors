"""Small tests of the graph update and provider used in the paper.

Author: Yanqi Li
"""
from __future__ import annotations

import itertools
from pathlib import Path
import sys
import unittest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from run import configure_paths

configure_paths()
from experiments.alpagd_rl.formal_20260810.pc3_v11_graph_recovery.run_graph_recovery import (
    initial_graph, TargetProvider, TARGETS,
)


class ProviderTests(unittest.TestCase):
    def test_reversal_threshold(self):
        graph = initial_graph("U", "P1")
        provider = TargetProvider(graph, mutable=True)
        for _ in range(8):
            provider.observe_first_hit("B", ["A"])
        provider.update()
        self.assertEqual(graph.probability(0, 1), 0.5)
        self.assertEqual(graph.probability(1, 0), 0.5)
        provider.observe_first_hit("B", ["A"])
        provider.update()
        self.assertGreater(graph.probability(0, 1), 0.5)
        self.assertLess(graph.probability(1, 0), 0.5)

    def test_frozen_graph_does_not_update(self):
        graph = initial_graph("F", "P2")
        before = graph.to_json()
        provider = TargetProvider(graph, mutable=False)
        provider.observe_first_hit("C", ["B"])
        self.assertFalse(provider.update()["changed"])
        self.assertEqual(graph.to_json(), before)

    def test_all_nonterminal_masks_have_normalized_distribution(self):
        provider = TargetProvider(initial_graph("O", "P1"), mutable=False)
        masks = [bits for bits in itertools.product((False, True), repeat=5) if not all(bits)]
        self.assertEqual(len(masks), 31)
        for bits in masks:
            achieved = dict(zip(TARGETS, bits))
            probabilities = provider.distribution(achieved, 0)
            self.assertAlmostEqual(sum(probabilities), 1.0)
            self.assertTrue(all(p >= 0 for p in probabilities))
            self.assertTrue(all(p == 0 for p, completed in zip(probabilities, bits) if completed))

    def test_correct_provider_selects_first_unfinished_target(self):
        provider = TargetProvider(initial_graph("O", "P1"), mutable=False)
        for depth in range(5):
            achieved = {name: index < depth for index, name in enumerate(TARGETS)}
            self.assertEqual(provider.select(achieved, 0, 0.3), TARGETS[depth])


if __name__ == "__main__":
    unittest.main()
