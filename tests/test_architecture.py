"""Architecture checks against the paper's equations and Table 1.

Run with:  python -m pytest tests/   (or: python tests/test_architecture.py)
"""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gam2 import DGMoEBNN                                   # noqa: E402
from gam2.models.bayesian_linear import BayesianLinear      # noqa: E402
from gam2.models.gating import RoutingGates                 # noqa: E402


class TestArchitecture(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)

    def test_default_hidden_size(self):
        # convenience default only: h = 16 for d <= 10, 64 for d > 10
        self.assertEqual(DGMoEBNN(10, 2).hidden_size, 16)
        self.assertEqual(DGMoEBNN(11, 2).hidden_size, 64)
        self.assertEqual(DGMoEBNN(32, 2).hidden_size, 64)

    def test_fusion_width_is_h_plus_s_minus_1(self):
        # Eq. (6): [q_1 h_1, ..., q_{s-1} h_{s-1}; h_base] -> h + s - 1 inputs
        for d, k in [(1, 2), (10, 2), (15, 3), (32, 4)]:
            m = DGMoEBNN(d, k)
            summary = m.architecture_summary()
            self.assertEqual(summary['hf_in'], m.hidden_size + k)
            self.assertEqual(summary['hf_in_expected'], m.hidden_size + k)

    def test_expert_input_width(self):
        # Eq. (2): expert receives [h_base ; r_t], each of width h
        m = DGMoEBNN(10, 3)
        self.assertEqual(m.architecture_summary()['expert_in'],
                         2 * m.hidden_size)

    def test_routing_gates_are_directed_pairs(self):
        # Eq. (3): one scalar per ordered pair (i, j), i != j
        m = DGMoEBNN(10, 3)
        self.assertEqual(m.architecture_summary()['routing_gates'], 3 * 2)
        _, _, _, _, alphas = m(torch.randn(4, 10))
        self.assertEqual(set(alphas.keys()),
                         {(i, j) for i in range(3) for j in range(3) if i != j})

    def test_routing_is_bidirectional_when_gates_open(self):
        # Eq. (4): with open gates the routed pass changes every expert, and the
        # result differs from the no-routing sweep (stage=1).
        m = DGMoEBNN(10, 2)
        with torch.no_grad():
            m.routing_gates.weight.fill_(2.0)          # sigmoid(2) > 0.5
        x = torch.randn(8, 10)
        mu_s1, _, _, _, _ = m(x, stage=1)
        mu_s2, _, _, _, _ = m(x, stage=2)
        self.assertFalse(torch.allclose(mu_s1, mu_s2, atol=1e-6))

    def test_routing_matches_direct_eq4(self):
        """Eq. (4) evaluated directly, with the gates open in both directions.

        The routed feature of expert t is compared with the explicit sum
        sum_{i != t} alpha_it h_i.  A single-pass implementation leaves
        r_t = 0 for t = 0 (no expert precedes it) and fails the first
        comparison, and the last check verifies that the second sweep really
        feeds the complete first-sweep feature set back into the experts.
        """
        torch.manual_seed(0)
        n = 3
        m = DGMoEBNN(6, n, hidden_size=12, expert_layers=1, hf_layers=1)
        with torch.no_grad():
            m.routing_gates.weight.fill_(3.0)          # every gate open
        x = torch.randn(7, 6)
        h_base = m.base_module(x)
        zeros = [torch.zeros_like(h_base) for _ in range(n)]
        _, feats = m._expert_pass(h_base, zeros)

        alpha = m.routing_gates.all_alphas()
        for t in range(n):
            expected = sum(alpha[(i, t)] * feats[i]
                           for i in range(n) if i != t)
            actual = m.routing_gates.routed_feature(feats, t)
            self.assertTrue(torch.allclose(expected, actual, atol=1e-6),
                            msg=f'expert {t} does not receive Eq. (4)')
        # the first expert receives information from the later experts
        self.assertGreater(
            float(m.routing_gates.routed_feature(feats, 0).abs().sum()), 0.0)

        # the routed pass reproduces the manual second sweep
        preds_manual, _ = m._expert_pass(h_base, feats)
        _, _, preds_model, _, _ = m(x, stage=2)
        for t in range(n):
            self.assertTrue(torch.allclose(preds_model[t], preds_manual[t],
                                           atol=1e-6))

    def test_closed_gates_make_routing_inactive(self):
        # gates initialised closed -> the routed pass is a no-op
        m = DGMoEBNN(10, 2)
        x = torch.randn(8, 10)
        mu_s1, _, _, _, _ = m(x, stage=1)
        mu_s2, _, _, _, _ = m(x, stage=2)
        self.assertTrue(torch.allclose(mu_s1, mu_s2, atol=1e-6))

    def test_quality_gate_is_a_softmax(self):
        # Eq. (5): q_t = softmax_t(f(h_t^-)) -> the gates sum to one
        m = DGMoEBNN(10, 3)
        _, _, _, q, _ = m(torch.randn(5, 10))
        total = sum(q_t for q_t in q)
        self.assertTrue(torch.allclose(total, torch.ones_like(total), atol=1e-6))

    def test_variance_is_non_negative(self):
        # Eq. (11): var = sum_j sigma_j^2 h_j^2 >= 0
        layer = BayesianLinear(6, 1)
        _, var = layer(torch.randn(4, 6))
        self.assertTrue((var >= 0).all())

    def test_backward_reaches_all_components(self):
        m = DGMoEBNN(10, 2)
        mu, _, _, _, _ = m(torch.randn(8, 10))
        loss = mu.pow(2).mean() + m.kl_divergence() * 1e-3
        loss.backward()
        for name, p in m.named_parameters():
            self.assertIsNotNone(p.grad, f'no gradient for {name}')

    def test_training_step(self):
        from gam2 import GAM2Trainer
        device = torch.device('cpu')
        m = DGMoEBNN(10, 2)
        data = [(torch.randn(16, 10), torch.randn(16, 1)),
                (torch.randn(12, 10), torch.randn(12, 1)),
                (torch.randn(6, 10), torch.randn(6, 1))]
        trainer = GAM2Trainer(m, base_lr=1e-3, lambda_gate=0.01)
        history = trainer.train(data, device, mode='e2e', e2e_epochs=2,
                                e2e_hf_weight=5.0, verbose=False)
        self.assertEqual(len(history['stage2']), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
