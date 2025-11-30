"""
Tests for loss functions.
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from Losses import (
    CrossEntropyLoss, BinaryCrossEntropyLoss, MeanSquaredError,
    MeanAbsoluteError, HuberLoss, PoissonLoss, CosineSimilarityLoss
)


class TestCrossEntropyLoss:
    def test_forward(self):
        loss_fn = CrossEntropyLoss()
        predictions = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        targets = np.array([[1, 0, 0], [0, 1, 0]])
        loss = loss_fn.forward(predictions, targets)
        assert loss > 0, "Loss should be positive"
        assert not np.isnan(loss), "Loss should not be NaN"
    
    def test_perfect_prediction(self):
        loss_fn = CrossEntropyLoss()
        predictions = np.array([[100.0, 0.0, 0.0]])
        targets = np.array([[1, 0, 0]])
        loss = loss_fn.forward(targets, predictions)
        assert loss < 0.1, f"Loss for perfect prediction should be near 0, got {loss}"
    
    def test_backward_shape(self):
        loss_fn = CrossEntropyLoss()
        predictions = np.array([[0.7, 0.2, 0.1]])
        targets = np.array([[1, 0, 0]])
        loss_fn.forward(predictions, targets)
        grad = loss_fn.backward()
        assert grad.shape == predictions.shape, f"Gradient shape mismatch: {grad.shape} vs {predictions.shape}"


class TestBinaryCrossEntropyLoss:
    def test_forward(self):
        loss_fn = BinaryCrossEntropyLoss()
        predictions = np.array([[0.9], [0.1]])
        targets = np.array([[1], [0]])
        loss = loss_fn.forward(predictions, targets)
        assert loss > 0, "Loss should be positive"
        assert not np.isnan(loss), "Loss should not be NaN"
    
    def test_numerical_stability(self):
        loss_fn = BinaryCrossEntropyLoss()
        predictions = np.array([[0.0], [1.0]])
        targets = np.array([[0], [1]])
        loss = loss_fn.forward(predictions, targets)
        assert not np.isnan(loss), "Loss should not be NaN for edge cases"
        assert not np.isinf(loss), "Loss should not be Inf for edge cases"


class TestMeanSquaredError:
    def test_forward(self):
        loss_fn = MeanSquaredError()
        predictions = np.array([[1.0, 2.0, 3.0]])
        targets = np.array([[1.0, 2.0, 3.0]])
        loss = loss_fn.forward(predictions, targets)
        assert np.allclose(loss, 0.0), f"Loss for identical values should be 0, got {loss}"
    
    def test_forward_difference(self):
        loss_fn = MeanSquaredError()
        predictions = np.array([[0.0]])
        targets = np.array([[1.0]])
        loss = loss_fn.forward(predictions, targets)
        assert np.allclose(loss, 1.0), f"Expected loss 1.0, got {loss}"
    
    def test_backward_shape(self):
        loss_fn = MeanSquaredError()
        predictions = np.array([[1.0, 2.0, 3.0]])
        targets = np.array([[0.0, 0.0, 0.0]])
        loss_fn.forward(predictions, targets)
        grad = loss_fn.backward()
        assert grad.shape == predictions.shape, f"Gradient shape mismatch"


class TestMeanAbsoluteError:
    def test_forward(self):
        loss_fn = MeanAbsoluteError()
        predictions = np.array([[1.0, 2.0, 3.0]])
        targets = np.array([[1.0, 2.0, 3.0]])
        loss = loss_fn.forward(predictions, targets)
        assert np.allclose(loss, 0.0), f"Loss for identical values should be 0, got {loss}"
    
    def test_forward_difference(self):
        loss_fn = MeanAbsoluteError()
        predictions = np.array([[0.0, 0.0]])
        targets = np.array([[1.0, -1.0]])
        loss = loss_fn.forward(predictions, targets)
        assert np.allclose(loss, 1.0), f"Expected loss 1.0, got {loss}"


class TestHuberLoss:
    def test_forward_small_error(self):
        loss_fn = HuberLoss(delta=1.0)
        predictions = np.array([[0.0]])
        targets = np.array([[0.5]])
        loss = loss_fn.forward(predictions, targets)
        expected = 0.5 * 0.5 ** 2
        assert np.allclose(loss, expected), f"Expected {expected}, got {loss}"
    
    def test_forward_large_error(self):
        loss_fn = HuberLoss(delta=1.0)
        predictions = np.array([[0.0]])
        targets = np.array([[2.0]])
        loss = loss_fn.forward(predictions, targets)
        expected = 1.0 * (2.0 - 0.5 * 1.0)
        assert np.allclose(loss, expected), f"Expected {expected}, got {loss}"


class TestPoissonLoss:
    def test_forward(self):
        loss_fn = PoissonLoss()
        predictions = np.array([[1.0, 2.0]])
        targets = np.array([[1.0, 2.0]])
        loss = loss_fn.forward(predictions, targets)
        assert not np.isnan(loss), "Loss should not be NaN"
    
    def test_numerical_stability(self):
        loss_fn = PoissonLoss()
        predictions = np.array([[0.001, 100.0]])
        targets = np.array([[1.0, 1.0]])
        loss = loss_fn.forward(predictions, targets)
        assert not np.isnan(loss), "Loss should not be NaN"
        assert not np.isinf(loss), "Loss should not be Inf"


class TestCosineSimilarityLoss:
    def test_forward_identical(self):
        loss_fn = CosineSimilarityLoss()
        predictions = np.array([[1.0, 0.0, 0.0]])
        targets = np.array([[1.0, 0.0, 0.0]])
        loss = loss_fn.forward(predictions, targets)
        assert np.allclose(loss, 0.0), f"Loss for identical vectors should be 0, got {loss}"
    
    def test_forward_orthogonal(self):
        loss_fn = CosineSimilarityLoss()
        predictions = np.array([[1.0, 0.0]])
        targets = np.array([[0.0, 1.0]])
        loss = loss_fn.forward(predictions, targets)
        assert np.allclose(loss, 1.0), f"Loss for orthogonal vectors should be 1, got {loss}"


def run_loss_tests():
    print("Testing Loss Functions...")
    
    tests = [
        ("CrossEntropy forward", TestCrossEntropyLoss().test_forward),
        ("CrossEntropy perfect", TestCrossEntropyLoss().test_perfect_prediction),
        ("CrossEntropy backward shape", TestCrossEntropyLoss().test_backward_shape),
        ("BinaryCE forward", TestBinaryCrossEntropyLoss().test_forward),
        ("BinaryCE stability", TestBinaryCrossEntropyLoss().test_numerical_stability),
        ("MSE forward zero", TestMeanSquaredError().test_forward),
        ("MSE forward difference", TestMeanSquaredError().test_forward_difference),
        ("MSE backward shape", TestMeanSquaredError().test_backward_shape),
        ("MAE forward zero", TestMeanAbsoluteError().test_forward),
        ("MAE forward difference", TestMeanAbsoluteError().test_forward_difference),
        ("Huber small error", TestHuberLoss().test_forward_small_error),
        ("Huber large error", TestHuberLoss().test_forward_large_error),
        ("Poisson forward", TestPoissonLoss().test_forward),
        ("Poisson stability", TestPoissonLoss().test_numerical_stability),
        ("Cosine identical", TestCosineSimilarityLoss().test_forward_identical),
        ("Cosine orthogonal", TestCosineSimilarityLoss().test_forward_orthogonal),
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
    
    print(f"\nLoss Tests: {passed} passed, {failed} failed")
    return passed, failed


if __name__ == "__main__":
    run_loss_tests()
