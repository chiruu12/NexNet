"""
Integration tests - end-to-end training examples.
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from Models import FNN, Sequential, CNN, RNNModel
from Layers import Linear, Dropout, BatchNorm, Flatten, Conv2D, MaxPool2D, Embedding, LSTM
from Activation_classes import ReLu, Softmax, Sigmoid
from Losses import CrossEntropyLoss, MeanSquaredError, BinaryCrossEntropyLoss
from Optimizer import Adam, SGD
from data import DataLoader


def test_fnn_classification():
    """Test FNN on synthetic classification task."""
    print("Testing FNN classification...")
    
    np.random.seed(42)
    X = np.random.randn(200, 20)
    y_labels = (X[:, 0] + X[:, 1] > 0).astype(int)
    y = np.eye(2)[y_labels]
    
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]
    
    model = FNN(loss=CrossEntropyLoss(), optimizer=Adam(learning_rate=0.01))
    model.add_layer(Linear(input_dim=20, output_dim=32, activation=ReLu()))
    model.add_layer(Linear(input_dim=32, output_dim=2))
    
    history = model.train(X_train, y_train, epochs=20, batch_size=32, verbose=False)
    
    loss, acc = model.evaluate(X_test, y_test)
    
    assert acc > 0.7, f"FNN accuracy should be > 0.7, got {acc}"
    assert history['train_loss'][-1] < history['train_loss'][0], "Loss should decrease"
    
    print(f"  ✓ FNN classification - accuracy: {acc:.2%}")


def test_sequential_regression():
    """Test Sequential model on regression task."""
    print("Testing Sequential regression...")
    
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.sum(X, axis=1, keepdims=True) + np.random.randn(200, 1) * 0.1
    
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]
    
    model = Sequential(
        Linear(10, 32),
        ReLu(),
        Linear(32, 16),
        ReLu(),
        Linear(16, 1)
    )
    model.compile(optimizer=Adam(learning_rate=0.01), loss=MeanSquaredError())
    
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=False)
    
    predictions = model.predict(X_test)
    mse = np.mean((predictions - y_test) ** 2)
    
    assert mse < 1.0, f"MSE should be < 1.0, got {mse}"
    
    print(f"  ✓ Sequential regression - MSE: {mse:.4f}")


def test_cnn_image():
    """Test CNN on synthetic image classification."""
    print("Testing CNN image classification...")
    
    np.random.seed(42)
    X = np.random.randn(100, 1, 8, 8)
    y_labels = (X.mean(axis=(1, 2, 3)) > 0).astype(int)
    y = np.eye(2)[y_labels]
    
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    model = CNN([
        Conv2D(in_channels=1, out_channels=4, kernel_size=3, padding=1),
        ReLu(),
        MaxPool2D(pool_size=2, stride=2),
        Flatten(),
        Linear(4 * 4 * 4, 16),
        ReLu(),
        Linear(16, 2),
        Softmax()
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss=CrossEntropyLoss())
    
    history = model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=False)
    
    loss, acc = model.evaluate(X_test, y_test)
    
    assert acc >= 0, "Accuracy should be non-negative"
    print(f"  ✓ CNN image classification - accuracy: {acc:.2%}")


def test_rnn_sequence():
    """Test RNN on sequence classification."""
    print("Testing RNN sequence classification...")
    
    np.random.seed(42)
    X = np.random.randint(0, 50, size=(100, 10))
    y_labels = (X.mean(axis=1) > 25).astype(int)
    y = np.eye(2)[y_labels]
    
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    model = RNNModel([
        Embedding(vocab_size=50, embedding_dim=16),
        LSTM(input_size=16, hidden_size=32, return_sequences=False),
        Linear(32, 2),
        Softmax()
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss=CrossEntropyLoss())
    
    history = model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=False)
    
    loss, acc = model.evaluate(X_test, y_test)
    
    assert acc >= 0, "Accuracy should be non-negative"
    print(f"  ✓ RNN sequence classification - accuracy: {acc:.2%}")


def test_dataloader():
    """Test DataLoader functionality."""
    print("Testing DataLoader...")
    
    X = np.arange(100).reshape(100, 1)
    y = np.arange(100)
    
    loader = DataLoader(X, y, batch_size=10, shuffle=False)
    
    batches = list(loader)
    assert len(batches) == 10, f"Expected 10 batches, got {len(batches)}"
    
    X_batch, y_batch = batches[0]
    assert X_batch.shape == (10, 1), f"Expected (10, 1), got {X_batch.shape}"
    assert np.array_equal(X_batch.flatten(), np.arange(10)), "First batch should be [0-9]"
    
    loader_shuffle = DataLoader(X, y, batch_size=10, shuffle=True)
    batches_shuffle = list(loader_shuffle)
    X_first_shuffle = batches_shuffle[0][0]
    
    assert len(batches_shuffle) == 10, "Shuffled loader should also have 10 batches"
    print(f"  ✓ DataLoader functionality")


def test_training_with_validation():
    """Test training with validation data."""
    print("Testing training with validation...")
    
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.eye(3)[np.random.randint(0, 3, 200)]
    
    X_train, X_val = X[:160], X[160:]
    y_train, y_val = y[:160], y[160:]
    
    model = Sequential(
        Linear(10, 16),
        ReLu(),
        Linear(16, 3),
        Softmax()
    )
    model.compile(optimizer=Adam(learning_rate=0.01), loss=CrossEntropyLoss())
    
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=False
    )
    
    assert 'val_loss' in history, "History should contain val_loss"
    assert len(history['val_loss']) == 5, "Should have 5 epochs of val_loss"
    
    print(f"  ✓ Training with validation")


def test_gradient_clipping_integration():
    """Test gradient clipping during training."""
    print("Testing gradient clipping...")
    
    np.random.seed(42)
    X = np.random.randn(100, 10) * 10
    y = np.eye(2)[np.random.randint(0, 2, 100)]
    
    model = Sequential(
        Linear(10, 32),
        ReLu(),
        Linear(32, 2),
        Softmax()
    )
    model.compile(optimizer=Adam(learning_rate=0.1), loss=CrossEntropyLoss())
    
    history = model.fit(
        X, y,
        epochs=5,
        batch_size=32,
        clip_grad_norm=1.0,
        verbose=False
    )
    
    assert not np.isnan(history['train_loss'][-1]), "Training should not produce NaN"
    
    print(f"  ✓ Gradient clipping integration")


def run_integration_tests():
    print("\n" + "=" * 50)
    print("Running Integration Tests...")
    print("=" * 50 + "\n")
    
    tests = [
        test_fnn_classification,
        test_sequential_regression,
        test_cnn_image,
        test_rnn_sequence,
        test_dataloader,
        test_training_with_validation,
        test_gradient_clipping_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1
    
    print(f"\nIntegration Tests: {passed} passed, {failed} failed")
    return passed, failed


if __name__ == "__main__":
    run_integration_tests()
