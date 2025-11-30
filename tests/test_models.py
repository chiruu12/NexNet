"""
Tests for model classes.
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from Models import FNN, Sequential, CNN, RNNModel, Transformer
from Layers import Linear, Dropout, BatchNorm, Flatten, Conv2D, MaxPool2D, Embedding, LSTM
from Activation_classes import ReLu, Softmax, Sigmoid
from Losses import CrossEntropyLoss, MeanSquaredError
from Optimizer import Adam, SGD


class TestFNN:
    def test_add_layer(self):
        model = FNN(loss=CrossEntropyLoss(), optimizer=Adam(learning_rate=0.001))
        model.add_layer(Linear(input_dim=10, output_dim=5))
        assert len(model.layers) == 1, "Layer not added"
    
    def test_forward(self):
        model = FNN(loss=CrossEntropyLoss(), optimizer=Adam(learning_rate=0.001))
        model.add_layer(Linear(input_dim=10, output_dim=5))
        model.add_layer(Softmax())
        
        x = np.random.randn(4, 10)
        output = model.forward(x)
        assert output.shape == (4, 5), f"Expected (4, 5), got {output.shape}"
    
    def test_train_eval_modes(self):
        model = FNN(loss=CrossEntropyLoss(), optimizer=Adam(learning_rate=0.001))
        model.add_layer(Linear(input_dim=10, output_dim=5))
        model.add_layer(Dropout(rate=0.5))
        
        model._set_training_mode(True)
        for layer in model.layers:
            if hasattr(layer, 'training'):
                assert layer.training == True, "Training mode not set on layer"
        
        model._set_training_mode(False)
        for layer in model.layers:
            if hasattr(layer, 'training'):
                assert layer.training == False, "Eval mode not set on layer"


class TestSequential:
    def test_init_with_layers(self):
        model = Sequential(
            Linear(10, 5),
            ReLu(),
            Linear(5, 2)
        )
        assert len(model.layers) == 3, f"Expected 3 layers, got {len(model.layers)}"
    
    def test_add(self):
        model = Sequential()
        model.add(Linear(10, 5))
        model.add(ReLu())
        assert len(model.layers) == 2, f"Expected 2 layers, got {len(model.layers)}"
    
    def test_forward(self):
        model = Sequential(
            Linear(10, 5),
            ReLu(),
            Linear(5, 2),
            Softmax()
        )
        x = np.random.randn(4, 10)
        output = model.forward(x)
        assert output.shape == (4, 2), f"Expected (4, 2), got {output.shape}"
        assert np.allclose(output.sum(axis=1), 1.0), "Softmax output doesn't sum to 1"
    
    def test_compile_and_fit(self):
        model = Sequential(
            Linear(10, 5),
            ReLu(),
            Linear(5, 3),
            Softmax()
        )
        model.compile(optimizer=SGD(learning_rate=0.01), loss=CrossEntropyLoss())
        
        X = np.random.randn(100, 10)
        y = np.eye(3)[np.random.randint(0, 3, 100)]
        
        history = model.fit(X, y, epochs=2, batch_size=32, verbose=False)
        
        assert 'train_loss' in history, "History should contain train_loss"
        assert len(history['train_loss']) == 2, "Should have 2 epochs of loss"
    
    def test_predict(self):
        model = Sequential(
            Linear(10, 5),
            ReLu(),
            Linear(5, 3),
            Softmax()
        )
        model.compile(optimizer=SGD(learning_rate=0.01), loss=CrossEntropyLoss())
        
        X = np.random.randn(20, 10)
        predictions = model.predict(X)
        
        assert predictions.shape == (20, 3), f"Expected (20, 3), got {predictions.shape}"
    
    def test_summary(self):
        model = Sequential(
            Linear(10, 5),
            ReLu(),
            Linear(5, 3)
        )
        model.summary()


class TestCNN:
    def test_init_with_layers(self):
        model = CNN([
            Conv2D(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            ReLu(),
            MaxPool2D(pool_size=2),
            Flatten(),
            Linear(8 * 14 * 14, 10)
        ])
        assert len(model.layers) == 5, f"Expected 5 layers, got {len(model.layers)}"
    
    def test_forward(self):
        model = CNN([
            Conv2D(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            ReLu(),
            Flatten(),
            Linear(8 * 28 * 28, 10),
            Softmax()
        ])
        x = np.random.randn(2, 1, 28, 28)
        output = model.forward(x)
        assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"


class TestRNNModel:
    def test_init_with_layers(self):
        model = RNNModel([
            Embedding(vocab_size=100, embedding_dim=32),
            LSTM(input_size=32, hidden_size=64, return_sequences=False),
            Linear(64, 10)
        ])
        assert len(model.layers) == 3, f"Expected 3 layers, got {len(model.layers)}"
    
    def test_forward(self):
        model = RNNModel([
            Embedding(vocab_size=100, embedding_dim=32),
            LSTM(input_size=32, hidden_size=64, return_sequences=False),
            Linear(64, 10),
            Softmax()
        ])
        x = np.random.randint(0, 100, size=(4, 20))
        output = model.forward(x)
        assert output.shape == (4, 10), f"Expected (4, 10), got {output.shape}"


class TestTransformer:
    def test_init(self):
        model = Transformer(
            vocab_size=1000,
            d_model=64,
            n_heads=4,
            n_layers=2,
            max_seq_len=128
        )
        assert model.vocab_size == 1000
        assert model.d_model == 64
        assert model.n_heads == 4
        assert model.n_layers == 2
    
    def test_forward(self):
        model = Transformer(
            vocab_size=1000,
            d_model=64,
            n_heads=4,
            n_layers=2,
            max_seq_len=128
        )
        x = np.random.randint(0, 1000, size=(2, 20))
        output = model.forward(x)
        assert output.shape == (2, 20, 1000), f"Expected (2, 20, 1000), got {output.shape}"
    
    def test_num_parameters(self):
        model = Transformer(
            vocab_size=1000,
            d_model=64,
            n_heads=4,
            n_layers=2,
            max_seq_len=128
        )
        num_params = model.num_parameters()
        assert num_params > 0, "Model should have parameters"


def run_model_tests():
    print("Testing Models...")
    
    tests = [
        ("FNN add layer", TestFNN().test_add_layer),
        ("FNN forward", TestFNN().test_forward),
        ("FNN train/eval modes", TestFNN().test_train_eval_modes),
        ("Sequential init", TestSequential().test_init_with_layers),
        ("Sequential add", TestSequential().test_add),
        ("Sequential forward", TestSequential().test_forward),
        ("Sequential compile & fit", TestSequential().test_compile_and_fit),
        ("Sequential predict", TestSequential().test_predict),
        ("Sequential summary", TestSequential().test_summary),
        ("CNN init", TestCNN().test_init_with_layers),
        ("CNN forward", TestCNN().test_forward),
        ("RNNModel init", TestRNNModel().test_init_with_layers),
        ("RNNModel forward", TestRNNModel().test_forward),
        ("Transformer init", TestTransformer().test_init),
        ("Transformer forward", TestTransformer().test_forward),
        ("Transformer params", TestTransformer().test_num_parameters),
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
    
    print(f"\nModel Tests: {passed} passed, {failed} failed")
    return passed, failed


if __name__ == "__main__":
    run_model_tests()
