"""
Tests for activation functions.
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from Activation_classes import ReLu, Sigmoid, Tanh, Softmax, LeakyReLu, ELU, PReLU, Swish, Softplus, GELU


class TestReLU:
    def test_forward_positive(self):
        relu = ReLu()
        x = np.array([[1.0, 2.0, 3.0]])
        output = relu.forward(x)
        expected = np.array([[1.0, 2.0, 3.0]])
        assert np.allclose(output, expected), f"Expected {expected}, got {output}"
    
    def test_forward_negative(self):
        relu = ReLu()
        x = np.array([[-1.0, -2.0, -3.0]])
        output = relu.forward(x)
        expected = np.array([[0.0, 0.0, 0.0]])
        assert np.allclose(output, expected), f"Expected {expected}, got {output}"
    
    def test_forward_mixed(self):
        relu = ReLu()
        x = np.array([[-1.0, 0.0, 1.0, 2.0]])
        output = relu.forward(x)
        expected = np.array([[0.0, 0.0, 1.0, 2.0]])
        assert np.allclose(output, expected), f"Expected {expected}, got {output}"
    
    def test_backward(self):
        relu = ReLu()
        x = np.array([[-1.0, 0.0, 1.0, 2.0]])
        relu.forward(x)
        grad = np.array([[1.0, 1.0, 1.0, 1.0]])
        output = relu.backward(grad)
        expected = np.array([[0.0, 0.0, 1.0, 1.0]])
        assert np.allclose(output, expected), f"Expected {expected}, got {output}"


class TestSigmoid:
    def test_forward(self):
        sigmoid = Sigmoid()
        x = np.array([[0.0]])
        output = sigmoid.forward(x)
        assert np.allclose(output, 0.5), f"Expected 0.5, got {output}"
    
    def test_forward_large_positive(self):
        sigmoid = Sigmoid()
        x = np.array([[100.0]])
        output = sigmoid.forward(x)
        assert np.allclose(output, 1.0, atol=1e-5), f"Expected ~1.0, got {output}"
    
    def test_forward_large_negative(self):
        sigmoid = Sigmoid()
        x = np.array([[-100.0]])
        output = sigmoid.forward(x)
        assert np.allclose(output, 0.0, atol=1e-5), f"Expected ~0.0, got {output}"
    
    def test_numerical_stability(self):
        sigmoid = Sigmoid()
        x = np.array([[1000.0, -1000.0]])
        output = sigmoid.forward(x)
        assert not np.isnan(output).any(), "Sigmoid produced NaN"
        assert not np.isinf(output).any(), "Sigmoid produced Inf"
    
    def test_backward(self):
        sigmoid = Sigmoid()
        x = np.array([[0.0]])
        sigmoid.forward(x)
        grad = np.array([[1.0]])
        output = sigmoid.backward(grad)
        expected = 0.25
        assert np.allclose(output, expected), f"Expected {expected}, got {output}"


class TestTanh:
    def test_forward(self):
        tanh = Tanh()
        x = np.array([[0.0]])
        output = tanh.forward(x)
        assert np.allclose(output, 0.0), f"Expected 0.0, got {output}"
    
    def test_range(self):
        tanh = Tanh()
        x = np.array([[-10.0, 0.0, 10.0]])
        output = tanh.forward(x)
        assert (output >= -1).all() and (output <= 1).all(), "Tanh output not in [-1, 1]"


class TestSoftmax:
    def test_forward_sums_to_one(self):
        softmax = Softmax()
        x = np.array([[1.0, 2.0, 3.0]])
        output = softmax.forward(x)
        assert np.allclose(output.sum(axis=1), 1.0), "Softmax doesn't sum to 1"
    
    def test_forward_positive(self):
        softmax = Softmax()
        x = np.array([[1.0, 2.0, 3.0]])
        output = softmax.forward(x)
        assert (output > 0).all(), "Softmax output should be positive"
    
    def test_numerical_stability(self):
        softmax = Softmax()
        x = np.array([[1000.0, 1001.0, 1002.0]])
        output = softmax.forward(x)
        assert not np.isnan(output).any(), "Softmax produced NaN"
        assert not np.isinf(output).any(), "Softmax produced Inf"


class TestLeakyReLU:
    def test_forward_positive(self):
        lrelu = LeakyReLu(alpha=0.01)
        x = np.array([[1.0, 2.0, 3.0]])
        output = lrelu.forward(x)
        expected = np.array([[1.0, 2.0, 3.0]])
        assert np.allclose(output, expected), f"Expected {expected}, got {output}"
    
    def test_forward_negative(self):
        lrelu = LeakyReLu(alpha=0.1)
        x = np.array([[-1.0, -2.0]])
        output = lrelu.forward(x)
        expected = np.array([[-0.1, -0.2]])
        assert np.allclose(output, expected), f"Expected {expected}, got {output}"


class TestELU:
    def test_forward_positive(self):
        elu = ELU(alpha=1.0)
        x = np.array([[1.0, 2.0, 3.0]])
        output = elu.forward(x)
        expected = np.array([[1.0, 2.0, 3.0]])
        assert np.allclose(output, expected), f"Expected {expected}, got {output}"
    
    def test_forward_negative(self):
        elu = ELU(alpha=1.0)
        x = np.array([[-1.0]])
        output = elu.forward(x)
        expected = np.exp(-1.0) - 1.0
        assert np.allclose(output, expected), f"Expected {expected}, got {output}"


class TestGELU:
    def test_forward_zero(self):
        gelu = GELU()
        x = np.array([[0.0]])
        output = gelu.forward(x)
        assert np.allclose(output, 0.0), f"Expected 0.0, got {output}"
    
    def test_forward_positive(self):
        gelu = GELU()
        x = np.array([[1.0, 2.0]])
        output = gelu.forward(x)
        assert (output > 0).all(), "GELU output should be positive for positive input"


def run_activation_tests():
    print("Testing Activation Functions...")
    
    tests = [
        ("ReLU forward positive", TestReLU().test_forward_positive),
        ("ReLU forward negative", TestReLU().test_forward_negative),
        ("ReLU forward mixed", TestReLU().test_forward_mixed),
        ("ReLU backward", TestReLU().test_backward),
        ("Sigmoid forward", TestSigmoid().test_forward),
        ("Sigmoid large positive", TestSigmoid().test_forward_large_positive),
        ("Sigmoid large negative", TestSigmoid().test_forward_large_negative),
        ("Sigmoid numerical stability", TestSigmoid().test_numerical_stability),
        ("Sigmoid backward", TestSigmoid().test_backward),
        ("Tanh forward", TestTanh().test_forward),
        ("Tanh range", TestTanh().test_range),
        ("Softmax sums to one", TestSoftmax().test_forward_sums_to_one),
        ("Softmax positive", TestSoftmax().test_forward_positive),
        ("Softmax numerical stability", TestSoftmax().test_numerical_stability),
        ("LeakyReLU positive", TestLeakyReLU().test_forward_positive),
        ("LeakyReLU negative", TestLeakyReLU().test_forward_negative),
        ("ELU positive", TestELU().test_forward_positive),
        ("ELU negative", TestELU().test_forward_negative),
        ("GELU zero", TestGELU().test_forward_zero),
        ("GELU positive", TestGELU().test_forward_positive),
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
    
    print(f"\nActivation Tests: {passed} passed, {failed} failed")
    return passed, failed


if __name__ == "__main__":
    run_activation_tests()
