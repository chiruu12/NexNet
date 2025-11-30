"""
Tests for layers.
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from Layers import Linear, Dropout, BatchNorm, Flatten, Conv2D, MaxPool2D, AvgPool2D
from Layers import RNN, LSTM, GRU, Embedding, LayerNorm


class TestLinear:
    def test_forward_shape(self):
        layer = Linear(input_dim=10, output_dim=5)
        x = np.random.randn(32, 10)
        output = layer.forward(x)
        assert output.shape == (32, 5), f"Expected (32, 5), got {output.shape}"
    
    def test_backward_shape(self):
        layer = Linear(input_dim=10, output_dim=5)
        x = np.random.randn(32, 10)
        layer.forward(x)
        grad = np.random.randn(32, 5)
        output = layer.backward(grad)
        assert output.shape == (32, 10), f"Expected (32, 10), got {output.shape}"
    
    def test_gradients_computed(self):
        layer = Linear(input_dim=10, output_dim=5)
        x = np.random.randn(32, 10)
        layer.forward(x)
        grad = np.random.randn(32, 5)
        layer.backward(grad)
        assert layer.dW is not None, "Weight gradient not computed"
        assert layer.db is not None, "Bias gradient not computed"
        assert layer.dW.shape == layer.W.shape, "Weight gradient shape mismatch"


class TestDropout:
    def test_forward_training(self):
        layer = Dropout(rate=0.5)
        layer.training = True
        x = np.ones((100, 100))
        output = layer.forward(x)
        zero_fraction = np.mean(output == 0)
        assert 0.3 < zero_fraction < 0.7, f"Dropout rate incorrect: {zero_fraction}"
    
    def test_forward_inference(self):
        layer = Dropout(rate=0.5)
        layer.training = False
        x = np.ones((100, 100))
        output = layer.forward(x)
        assert np.allclose(output, x), "Dropout should pass through during inference"
    
    def test_scaling(self):
        layer = Dropout(rate=0.5)
        layer.training = True
        x = np.ones((1000, 1000))
        output = layer.forward(x)
        mean_output = np.mean(output)
        assert 0.8 < mean_output < 1.2, f"Dropout scaling incorrect: mean={mean_output}"


class TestBatchNorm:
    def test_forward_shape(self):
        layer = BatchNorm(num_features=64)
        x = np.random.randn(32, 64)
        output = layer.forward(x)
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    
    def test_normalization_training(self):
        layer = BatchNorm(num_features=64)
        layer.training = True
        x = np.random.randn(32, 64) * 5 + 10
        output = layer.forward(x)
        mean = np.mean(output, axis=0)
        std = np.std(output, axis=0)
        assert np.allclose(mean, 0, atol=0.1), f"Mean not near 0: {np.mean(mean)}"
        assert np.allclose(std, 1, atol=0.1), f"Std not near 1: {np.mean(std)}"


class TestFlatten:
    def test_forward_shape(self):
        layer = Flatten()
        x = np.random.randn(32, 3, 28, 28)
        output = layer.forward(x)
        assert output.shape == (32, 3 * 28 * 28), f"Expected (32, 2352), got {output.shape}"
    
    def test_backward_shape(self):
        layer = Flatten()
        x = np.random.randn(32, 3, 28, 28)
        layer.forward(x)
        grad = np.random.randn(32, 3 * 28 * 28)
        output = layer.backward(grad)
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"


class TestConv2D:
    def test_forward_shape(self):
        layer = Conv2D(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        x = np.random.randn(4, 1, 28, 28)
        output = layer.forward(x)
        assert output.shape == (4, 32, 28, 28), f"Expected (4, 32, 28, 28), got {output.shape}"
    
    def test_forward_no_padding(self):
        layer = Conv2D(in_channels=1, out_channels=16, kernel_size=3, padding=0)
        x = np.random.randn(4, 1, 28, 28)
        output = layer.forward(x)
        assert output.shape == (4, 16, 26, 26), f"Expected (4, 16, 26, 26), got {output.shape}"
    
    def test_forward_stride(self):
        layer = Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)
        x = np.random.randn(4, 1, 28, 28)
        output = layer.forward(x)
        assert output.shape == (4, 16, 14, 14), f"Expected (4, 16, 14, 14), got {output.shape}"


class TestPooling:
    def test_maxpool_forward_shape(self):
        layer = MaxPool2D(pool_size=2, stride=2)
        x = np.random.randn(4, 32, 28, 28)
        output = layer.forward(x)
        assert output.shape == (4, 32, 14, 14), f"Expected (4, 32, 14, 14), got {output.shape}"
    
    def test_avgpool_forward_shape(self):
        layer = AvgPool2D(pool_size=2, stride=2)
        x = np.random.randn(4, 32, 28, 28)
        output = layer.forward(x)
        assert output.shape == (4, 32, 14, 14), f"Expected (4, 32, 14, 14), got {output.shape}"
    
    def test_maxpool_values(self):
        layer = MaxPool2D(pool_size=2, stride=2)
        x = np.array([[[[1, 2], [3, 4]]]]).astype(float)
        output = layer.forward(x)
        assert output[0, 0, 0, 0] == 4, f"MaxPool should return max value"


class TestRNN:
    def test_forward_shape(self):
        layer = RNN(input_size=32, hidden_size=64, return_sequences=True)
        x = np.random.randn(4, 10, 32)
        output = layer.forward(x)
        assert output.shape == (4, 10, 64), f"Expected (4, 10, 64), got {output.shape}"
    
    def test_forward_last_only(self):
        layer = RNN(input_size=32, hidden_size=64, return_sequences=False)
        x = np.random.randn(4, 10, 32)
        output = layer.forward(x)
        assert output.shape == (4, 64), f"Expected (4, 64), got {output.shape}"


class TestLSTM:
    def test_forward_shape(self):
        layer = LSTM(input_size=32, hidden_size=64, return_sequences=True)
        x = np.random.randn(4, 10, 32)
        output = layer.forward(x)
        assert output.shape == (4, 10, 64), f"Expected (4, 10, 64), got {output.shape}"
    
    def test_forward_last_only(self):
        layer = LSTM(input_size=32, hidden_size=64, return_sequences=False)
        x = np.random.randn(4, 10, 32)
        output = layer.forward(x)
        assert output.shape == (4, 64), f"Expected (4, 64), got {output.shape}"


class TestGRU:
    def test_forward_shape(self):
        layer = GRU(input_size=32, hidden_size=64, return_sequences=True)
        x = np.random.randn(4, 10, 32)
        output = layer.forward(x)
        assert output.shape == (4, 10, 64), f"Expected (4, 10, 64), got {output.shape}"


class TestEmbedding:
    def test_forward_shape(self):
        layer = Embedding(vocab_size=1000, embedding_dim=128)
        x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        output = layer.forward(x)
        assert output.shape == (2, 5, 128), f"Expected (2, 5, 128), got {output.shape}"
    
    def test_forward_values(self):
        layer = Embedding(vocab_size=100, embedding_dim=32)
        x = np.array([[0, 0]])
        output = layer.forward(x)
        assert np.allclose(output[0, 0], output[0, 1]), "Same token should have same embedding"


class TestLayerNorm:
    def test_forward_shape(self):
        layer = LayerNorm(normalized_shape=64)
        x = np.random.randn(32, 10, 64)
        output = layer.forward(x)
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    
    def test_normalization(self):
        layer = LayerNorm(normalized_shape=64)
        x = np.random.randn(32, 10, 64) * 5 + 10
        output = layer.forward(x)
        mean = np.mean(output, axis=-1)
        std = np.std(output, axis=-1)
        assert np.allclose(mean, 0, atol=0.1), f"Mean not near 0"
        assert np.allclose(std, 1, atol=0.1), f"Std not near 1"


def run_layer_tests():
    print("Testing Layers...")
    
    tests = [
        ("Linear forward shape", TestLinear().test_forward_shape),
        ("Linear backward shape", TestLinear().test_backward_shape),
        ("Linear gradients computed", TestLinear().test_gradients_computed),
        ("Dropout training", TestDropout().test_forward_training),
        ("Dropout inference", TestDropout().test_forward_inference),
        ("Dropout scaling", TestDropout().test_scaling),
        ("BatchNorm shape", TestBatchNorm().test_forward_shape),
        ("BatchNorm normalization", TestBatchNorm().test_normalization_training),
        ("Flatten forward", TestFlatten().test_forward_shape),
        ("Flatten backward", TestFlatten().test_backward_shape),
        ("Conv2D forward shape", TestConv2D().test_forward_shape),
        ("Conv2D no padding", TestConv2D().test_forward_no_padding),
        ("Conv2D stride", TestConv2D().test_forward_stride),
        ("MaxPool shape", TestPooling().test_maxpool_forward_shape),
        ("AvgPool shape", TestPooling().test_avgpool_forward_shape),
        ("MaxPool values", TestPooling().test_maxpool_values),
        ("RNN sequences", TestRNN().test_forward_shape),
        ("RNN last only", TestRNN().test_forward_last_only),
        ("LSTM sequences", TestLSTM().test_forward_shape),
        ("LSTM last only", TestLSTM().test_forward_last_only),
        ("GRU sequences", TestGRU().test_forward_shape),
        ("Embedding shape", TestEmbedding().test_forward_shape),
        ("Embedding values", TestEmbedding().test_forward_values),
        ("LayerNorm shape", TestLayerNorm().test_forward_shape),
        ("LayerNorm normalization", TestLayerNorm().test_normalization),
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
    
    print(f"\nLayer Tests: {passed} passed, {failed} failed")
    return passed, failed


if __name__ == "__main__":
    run_layer_tests()
