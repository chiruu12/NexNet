"""
Tests for utilities (gradient clipping, regularization, etc.)
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from utils import clip_grad_norm, clip_grad_value
from utils import L1Regularization, L2Regularization, ElasticNetRegularization
from utils import WeightDecay, MaxNormConstraint, UnitNormConstraint
from Layers import Linear


class TestGradientClipping:
    def test_clip_grad_norm(self):
        layer = Linear(input_dim=10, output_dim=5)
        layer.dW = np.ones_like(layer.W) * 10
        layer.db = np.ones_like(layer.b) * 10
        
        total_norm = clip_grad_norm([layer], max_norm=1.0)
        
        new_norm = np.sqrt(np.sum(layer.dW ** 2) + np.sum(layer.db ** 2))
        assert new_norm <= 1.0 + 1e-6, f"Gradient norm should be <= 1.0, got {new_norm}"
    
    def test_clip_grad_norm_no_clip_needed(self):
        layer = Linear(input_dim=10, output_dim=5)
        layer.dW = np.ones_like(layer.W) * 0.01
        layer.db = np.ones_like(layer.b) * 0.01
        original_dW = layer.dW.copy()
        
        clip_grad_norm([layer], max_norm=10.0)
        
        assert np.allclose(layer.dW, original_dW), "Gradients should not change if norm < max_norm"
    
    def test_clip_grad_value(self):
        layer = Linear(input_dim=10, output_dim=5)
        layer.dW = np.ones_like(layer.W) * 100
        layer.db = np.ones_like(layer.b) * -100
        
        clip_grad_value([layer], clip_value=1.0)
        
        assert np.all(layer.dW <= 1.0), "dW should be clipped to <= 1.0"
        assert np.all(layer.dW >= -1.0), "dW should be clipped to >= -1.0"
        assert np.all(layer.db <= 1.0), "db should be clipped to <= 1.0"
        assert np.all(layer.db >= -1.0), "db should be clipped to >= -1.0"


class TestL1Regularization:
    def test_loss(self):
        reg = L1Regularization(lambda_reg=0.01)
        layer = Linear(input_dim=10, output_dim=5)
        layer.W = np.ones_like(layer.W)
        
        loss = reg.loss([layer])
        expected = 0.01 * np.sum(np.abs(layer.W))
        
        assert np.allclose(loss, expected), f"Expected {expected}, got {loss}"
    
    def test_apply_gradients(self):
        reg = L1Regularization(lambda_reg=0.01)
        layer = Linear(input_dim=10, output_dim=5)
        layer.W = np.ones_like(layer.W)
        layer.dW = np.zeros_like(layer.W)
        
        reg.apply_gradients([layer])
        
        expected_grad = 0.01 * np.sign(layer.W)
        assert np.allclose(layer.dW, expected_grad), "L1 gradient not applied correctly"


class TestL2Regularization:
    def test_loss(self):
        reg = L2Regularization(lambda_reg=0.01)
        layer = Linear(input_dim=10, output_dim=5)
        layer.W = np.ones_like(layer.W) * 2
        
        loss = reg.loss([layer])
        expected = 0.5 * 0.01 * np.sum(layer.W ** 2)
        
        assert np.allclose(loss, expected), f"Expected {expected}, got {loss}"
    
    def test_apply_gradients(self):
        reg = L2Regularization(lambda_reg=0.01)
        layer = Linear(input_dim=10, output_dim=5)
        layer.W = np.ones_like(layer.W) * 2
        layer.dW = np.zeros_like(layer.W)
        
        reg.apply_gradients([layer])
        
        expected_grad = 0.01 * layer.W
        assert np.allclose(layer.dW, expected_grad), "L2 gradient not applied correctly"


class TestElasticNetRegularization:
    def test_loss(self):
        reg = ElasticNetRegularization(alpha=0.01, l1_ratio=0.5)
        layer = Linear(input_dim=10, output_dim=5)
        layer.W = np.ones_like(layer.W)
        
        loss = reg.loss([layer])
        expected_l1 = np.sum(np.abs(layer.W))
        expected_l2 = np.sum(layer.W ** 2)
        expected = 0.01 * (0.5 * expected_l1 + 0.5 * 0.5 * expected_l2)
        
        assert np.allclose(loss, expected), f"Expected {expected}, got {loss}"


class TestWeightDecay:
    def test_apply(self):
        decay = WeightDecay(decay=0.1)
        layer = Linear(input_dim=10, output_dim=5)
        layer.W = np.ones_like(layer.W)
        initial_W = layer.W.copy()
        
        decay.apply([layer], learning_rate=1.0)
        
        expected = initial_W - 0.1 * initial_W
        assert np.allclose(layer.W, expected), "Weight decay not applied correctly"


class TestMaxNormConstraint:
    def test_apply(self):
        constraint = MaxNormConstraint(max_norm=1.0, axis=0)
        layer = Linear(input_dim=10, output_dim=5)
        layer.W = np.ones_like(layer.W) * 10
        
        constraint.apply([layer])
        
        norms = np.linalg.norm(layer.W, axis=0)
        assert np.all(norms <= 1.0 + 1e-6), f"Norms should be <= 1.0, got max {norms.max()}"


class TestUnitNormConstraint:
    def test_apply(self):
        constraint = UnitNormConstraint(axis=0)
        layer = Linear(input_dim=10, output_dim=5)
        layer.W = np.random.randn(10, 5) * 10
        
        constraint.apply([layer])
        
        norms = np.linalg.norm(layer.W, axis=0)
        assert np.allclose(norms, 1.0), f"Norms should be 1.0, got {norms}"


def run_utils_tests():
    print("Testing Utilities...")
    
    tests = [
        ("clip_grad_norm", TestGradientClipping().test_clip_grad_norm),
        ("clip_grad_norm no clip", TestGradientClipping().test_clip_grad_norm_no_clip_needed),
        ("clip_grad_value", TestGradientClipping().test_clip_grad_value),
        ("L1 loss", TestL1Regularization().test_loss),
        ("L1 gradients", TestL1Regularization().test_apply_gradients),
        ("L2 loss", TestL2Regularization().test_loss),
        ("L2 gradients", TestL2Regularization().test_apply_gradients),
        ("ElasticNet loss", TestElasticNetRegularization().test_loss),
        ("WeightDecay apply", TestWeightDecay().test_apply),
        ("MaxNorm constraint", TestMaxNormConstraint().test_apply),
        ("UnitNorm constraint", TestUnitNormConstraint().test_apply),
    ]
    
    passed = 0
    failed = 0
    
    for name, test in tests:
        try:
            test()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
    
    print(f"\nUtils Tests: {passed} passed, {failed} failed")
    return passed, failed


if __name__ == "__main__":
    run_utils_tests()
