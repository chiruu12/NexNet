"""
Tests for optimizers.
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from Optimizer import SGD, Momentum, AdaGrad, RMSProp, AdaDelta, Adam, AdamW, NAdam
from Layers import Linear


class TestSGD:
    def test_step(self):
        optimizer = SGD(learning_rate=0.1)
        layer = Linear(input_dim=10, output_dim=5)
        initial_W = layer.W.copy()
        
        x = np.random.randn(4, 10)
        layer.forward(x)
        grad = np.random.randn(4, 5)
        layer.backward(grad)
        
        optimizer.step([layer])
        
        assert not np.allclose(layer.W, initial_W), "Weights should have changed"
    
    def test_learning_rate(self):
        optimizer = SGD(learning_rate=0.01)
        assert optimizer.learning_rate == 0.01, "Learning rate not set correctly"


class TestMomentum:
    def test_step(self):
        optimizer = Momentum(learning_rate=0.1, momentum=0.9)
        layer = Linear(input_dim=10, output_dim=5)
        initial_W = layer.W.copy()
        
        x = np.random.randn(4, 10)
        layer.forward(x)
        grad = np.random.randn(4, 5)
        layer.backward(grad)
        
        optimizer.step([layer])
        
        assert not np.allclose(layer.W, initial_W), "Weights should have changed"
    
    def test_velocity_initialization(self):
        optimizer = Momentum(learning_rate=0.1, momentum=0.9)
        layer = Linear(input_dim=10, output_dim=5)
        
        x = np.random.randn(4, 10)
        layer.forward(x)
        grad = np.random.randn(4, 5)
        layer.backward(grad)
        
        optimizer.step([layer])
        
        assert len(optimizer.v_W) > 0, "Velocity should be initialized"


class TestAdam:
    def test_step(self):
        optimizer = Adam(learning_rate=0.001)
        layer = Linear(input_dim=10, output_dim=5)
        initial_W = layer.W.copy()
        
        x = np.random.randn(4, 10)
        layer.forward(x)
        grad = np.random.randn(4, 5)
        layer.backward(grad)
        
        optimizer.step([layer])
        
        assert not np.allclose(layer.W, initial_W), "Weights should have changed"
    
    def test_moment_initialization(self):
        optimizer = Adam(learning_rate=0.001)
        layer = Linear(input_dim=10, output_dim=5)
        
        x = np.random.randn(4, 10)
        layer.forward(x)
        grad = np.random.randn(4, 5)
        layer.backward(grad)
        
        optimizer.step([layer])
        
        assert len(optimizer.m_W) > 0, "First moment should be initialized"
        assert len(optimizer.v_W) > 0, "Second moment should be initialized"


class TestAdamW:
    def test_weight_decay(self):
        optimizer = AdamW(learning_rate=0.001, weight_decay=0.01)
        layer = Linear(input_dim=10, output_dim=5)
        layer.W = np.ones_like(layer.W)
        initial_W = layer.W.copy()
        
        x = np.random.randn(4, 10)
        layer.forward(x)
        layer.dW = np.zeros_like(layer.W)
        layer.db = np.zeros_like(layer.b)
        
        optimizer.step([layer])
        
        assert np.all(layer.W < initial_W), "Weight decay should reduce weights"


class TestRMSProp:
    def test_step(self):
        optimizer = RMSProp(learning_rate=0.01)
        layer = Linear(input_dim=10, output_dim=5)
        initial_W = layer.W.copy()
        
        x = np.random.randn(4, 10)
        layer.forward(x)
        grad = np.random.randn(4, 5)
        layer.backward(grad)
        
        optimizer.step([layer])
        
        assert not np.allclose(layer.W, initial_W), "Weights should have changed"


class TestAdaGrad:
    def test_step(self):
        optimizer = AdaGrad(learning_rate=0.1)
        layer = Linear(input_dim=10, output_dim=5)
        initial_W = layer.W.copy()
        
        x = np.random.randn(4, 10)
        layer.forward(x)
        grad = np.random.randn(4, 5)
        layer.backward(grad)
        
        optimizer.step([layer])
        
        assert not np.allclose(layer.W, initial_W), "Weights should have changed"


class TestAdaDelta:
    def test_step(self):
        optimizer = AdaDelta()
        layer = Linear(input_dim=10, output_dim=5)
        initial_W = layer.W.copy()
        
        x = np.random.randn(4, 10)
        layer.forward(x)
        grad = np.random.randn(4, 5)
        layer.backward(grad)
        
        optimizer.step([layer])
        
        assert not np.allclose(layer.W, initial_W), "Weights should have changed"


class TestNAdam:
    def test_step(self):
        optimizer = NAdam(learning_rate=0.001)
        layer = Linear(input_dim=10, output_dim=5)
        initial_W = layer.W.copy()
        
        x = np.random.randn(4, 10)
        layer.forward(x)
        grad = np.random.randn(4, 5)
        layer.backward(grad)
        
        optimizer.step([layer])
        
        assert not np.allclose(layer.W, initial_W), "Weights should have changed"


class TestOptimizerMultipleLayers:
    def test_multiple_layers(self):
        optimizer = Adam(learning_rate=0.001)
        layers = [
            Linear(input_dim=10, output_dim=20),
            Linear(input_dim=20, output_dim=5)
        ]
        initial_weights = [l.W.copy() for l in layers]
        
        x = np.random.randn(4, 10)
        out = layers[0].forward(x)
        out = layers[1].forward(out)
        
        grad = np.random.randn(4, 5)
        grad = layers[1].backward(grad)
        grad = layers[0].backward(grad)
        
        optimizer.step(layers)
        
        for i, layer in enumerate(layers):
            assert not np.allclose(layer.W, initial_weights[i]), f"Layer {i} weights should have changed"


def run_optimizer_tests():
    print("Testing Optimizers...")
    
    tests = [
        ("SGD step", TestSGD().test_step),
        ("SGD learning rate", TestSGD().test_learning_rate),
        ("Momentum step", TestMomentum().test_step),
        ("Momentum velocity init", TestMomentum().test_velocity_initialization),
        ("Adam step", TestAdam().test_step),
        ("Adam moment init", TestAdam().test_moment_initialization),
        ("AdamW weight decay", TestAdamW().test_weight_decay),
        ("RMSProp step", TestRMSProp().test_step),
        ("AdaGrad step", TestAdaGrad().test_step),
        ("AdaDelta step", TestAdaDelta().test_step),
        ("NAdam step", TestNAdam().test_step),
        ("Multiple layers", TestOptimizerMultipleLayers().test_multiple_layers),
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
    
    print(f"\nOptimizer Tests: {passed} passed, {failed} failed")
    return passed, failed


if __name__ == "__main__":
    run_optimizer_tests()
