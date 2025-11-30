# NexNet ⚙️

NexNet is a neural network framework implemented from scratch using NumPy. It provides functionalities similar to PyTorch and TensorFlow, including various activation functions, loss functions, optimizers, and more. This README will guide you through the setup, usage, and features of the framework.

## Features

### Model Classes

- `FNN`: Feedforward Neural Network for classification and regression.
- `Sequential`: PyTorch-like sequential container for building models layer by layer.
- `CNN`: Convolutional Neural Network model for image tasks.
- `RNNModel`: Recurrent Neural Network model for sequence tasks.
- `Transformer`: GPT-style transformer model for language modeling.

### Activation Functions

- `ReLU`: Rectified Linear Unit, introduces non-linearity by zeroing out negative values.
- `Softmax`: Converts logits to probabilities, commonly used in the output layer for classification tasks.
- `PReLU`: Parametric ReLU, allows for a learnable slope for negative values.
- `Sigmoid`: Maps values to a range between 0 and 1, often used in binary classification.
- `Tanh`: Maps values to a range between -1 and 1, helping with centering data.
- `LeakyReLU`: Similar to ReLU but allows a small gradient when inputs are negative.
- `ELU`: Exponential Linear Unit, helps speed up learning by smoothing the activation function.
- `Swish`: Smooth, non-monotonic activation function that can improve model performance.
- `Softplus`: A smooth approximation to ReLU, improving gradient flow.
- `GELU`: Gaussian Error Linear Unit, used in GPT/BERT transformer models.

### Loss Functions

- `CrossEntropyLoss`: For multi-class classification with built-in softmax.
- `BinaryCrossEntropyLoss`: For binary classification tasks.
- `MSE`: Mean Squared Error for regression tasks.
- `MAE`: Mean Absolute Error for regression tasks.
- `HuberLoss`: Combines MSE and MAE advantages, robust to outliers.
- `PoissonLoss`: For count-based prediction tasks.
- `CosineSimilarityLoss`: Measures angular distance between vectors.

### Optimizers

- `SGD`: Stochastic Gradient Descent.
- `Momentum`: SGD with momentum for faster convergence.
- `AdaGrad`: Adaptive learning rates based on gradient history.
- `RMSProp`: Adaptive learning rates with moving average.
- `AdaDelta`: Extension of AdaGrad with reduced learning rate decay.
- `Adam`: Adaptive moment estimation, combines AdaGrad and RMSProp.
- `AdamW`: Adam with decoupled weight decay regularization.
- `NAdam`: Adam with Nesterov momentum.

### Layers

#### Dense Layers

- `Linear`: Fully connected layer with optional activation function.

#### CNN Layers

- `Conv2D`: 2D convolutional layer with configurable kernel, stride, and padding.
- `MaxPool2D`: Max pooling layer for spatial downsampling.
- `AvgPool2D`: Average pooling layer for smooth downsampling.

#### RNN Layers

- `RNN`: Vanilla recurrent neural network for sequence processing.
- `LSTM`: Long Short-Term Memory for learning long-term dependencies.
- `GRU`: Gated Recurrent Unit, a simpler alternative to LSTM.
- `Embedding`: Converts integer indices to dense vectors for NLP tasks.

#### Transformer Layers (GPT/BERT)

- `MultiHeadAttention`: Multi-head self-attention mechanism.
- `ScaledDotProductAttention`: Core attention operation with masking support.
- `TransformerDecoderBlock`: GPT-style decoder block with causal masking.
- `TransformerEncoderBlock`: BERT-style encoder block.
- `FeedForward`: Position-wise feed-forward network.
- `LayerNorm`: Layer normalization (different from BatchNorm).
- `SinusoidalPositionalEncoding`: Fixed positional encoding from "Attention Is All You Need".
- `LearnedPositionalEncoding`: Learnable position embeddings (GPT/BERT style).

#### Regularization & Utility Layers

- `Dropout`: Regularization layer that randomly drops units during training.
- `BatchNorm`: Batch normalization for faster and more stable training.
- `Flatten`: Reshapes input to 2D for transition to fully connected layers.

### Regularization

- `L1Regularization`: Lasso regularization for sparse weights.
- `L2Regularization`: Ridge regularization for small weights.
- `ElasticNetRegularization`: Combination of L1 and L2.
- `WeightDecay`: Direct weight decay during optimization.
- `MaxNormConstraint`: Clip weights by max norm.
- `UnitNormConstraint`: Normalize weights to unit norm.

### Gradient Utilities

- `clip_grad_norm`: Clip gradient norm to prevent exploding gradients.
- `clip_grad_value`: Clip gradient values to a range.

### Initializers

- `Xavier`: For sigmoid and tanh activations.
- `He`: For ReLU-based activations.
- `Random`: Simple random initialization.
- `Zero`: Zero initialization.

### Learning Rate Schedulers

- `StepLR`: Decay learning rate by factor every N epochs.
- `ExponentialLR`: Exponential decay every epoch.
- `CosineAnnealingLR`: Cosine annealing schedule.
- `ReduceLROnPlateau`: Reduce LR when metric stops improving.

### Callbacks

- `EarlyStopping`: Stop training when metric stops improving.
- `ModelCheckpoint`: Save model when metric improves.
- `History`: Record and plot training history.

### Metrics

- `accuracy`: Classification accuracy.
- `precision`: Precision score with averaging options.
- `recall`: Recall score with averaging options.
- `f1_score`: F1 score (harmonic mean of precision and recall).
- `confusion_matrix`: Confusion matrix for classification.
- `mean_squared_error`: MSE metric for regression.
- `mean_absolute_error`: MAE metric for regression.
- `r2_score`: R-squared coefficient of determination.

### Data Utilities

- `DataLoader`: Batch data loading with shuffling support.
- `train_test_split`: Split data into train and test sets.
- `OneHotEncoder`: Encode/decode one-hot vectors.

### Core Classes (PyTorch-like)

- `Module`: Base class for all modules (similar to nn.Module).
- `Parameter`: Wrapper for trainable parameters.
- `init_weights`: Weight initialization utility function.

## Installation

Clone the repository:

```bash
git clone https://github.com/chiruu12/NexNet.git
cd NexNet
pip install -r requirements.txt
```

## Usage

### Importing Modules

```python
from Models import FNN, Sequential, CNN, RNNModel, Transformer
from Losses import CrossEntropyLoss, MSE, MAE, HuberLoss
from Layers import Linear, Dropout, BatchNorm, Flatten, Conv2D, MaxPool2D, RNN, LSTM, GRU, Embedding
from Activation_classes import ReLu, Softmax, PReLU, Sigmoid, Tanh, LeakyReLu, ELU, Swish, Softplus, GELU
from utils import OneHotEncoder, Initializer, clip_grad_norm, L1Regularization, L2Regularization
from Optimizer import SGD, Momentum, AdaGrad, Adam, AdamW, NAdam, RMSProp, AdaDelta
from schedulers import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from callbacks import EarlyStopping, ModelCheckpoint, History
from metrics import accuracy, precision, recall, f1_score, confusion_matrix
from data import DataLoader, train_test_split
from core import Module, Parameter, init_weights
```

### Data Preparation

```python
from data import train_test_split
from utils import OneHotEncoder

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

encoder = OneHotEncoder(num_classes=10)
y_train_encoded = encoder.encode(y_train)
y_test_encoded = encoder.encode(y_test)
```

### Creating and Training a Model

```python
from Models import FNN
from Layers import Linear, Dropout, BatchNorm
from Activation_classes import ReLu
from Losses import CrossEntropyLoss
from Optimizer import Adam

model = FNN(loss=CrossEntropyLoss(), optimizer=Adam(learning_rate=0.001))

model.add_layer(Linear(input_dim=784, output_dim=256, activation=ReLu()))
model.add_layer(BatchNorm(256))
model.add_layer(Dropout(rate=0.3))
model.add_layer(Linear(input_dim=256, output_dim=128, activation=ReLu()))
model.add_layer(Dropout(rate=0.3))
model.add_layer(Linear(input_dim=128, output_dim=10))

model.summary()

history = model.train(
    X_train, y_train_encoded,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    verbose=True
)

loss, accuracy = model.evaluate(X_test, y_test_encoded)
```

### PyTorch-like Sequential Model

```python
from Models import Sequential
from Layers import Linear, Dropout, BatchNorm
from Activation_classes import ReLu, Softmax
from Losses import CrossEntropyLoss
from Optimizer import Adam

# Build model using Sequential container
model = Sequential(
    Linear(784, 256),
    ReLu(),
    BatchNorm(256),
    Dropout(rate=0.3),
    Linear(256, 128),
    ReLu(),
    Dropout(rate=0.3),
    Linear(128, 10),
    Softmax()
)

# Compile and train
model.compile(optimizer=Adam(learning_rate=0.001), loss=CrossEntropyLoss())
history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_val, y_val))

# Or build incrementally
model = Sequential()
model.add(Linear(784, 256))
model.add(ReLu())
model.add(Linear(256, 10))
```

### CNN Model (Image Classification)

```python
from Models import CNN
from Layers import Conv2D, MaxPool2D, Flatten, Linear, Dropout
from Activation_classes import ReLu, Softmax
from Losses import CrossEntropyLoss
from Optimizer import Adam

# Build CNN for MNIST
model = CNN([
    Conv2D(in_channels=1, out_channels=32, kernel_size=3, padding=1),
    ReLu(),
    MaxPool2D(pool_size=2, stride=2),
    Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1),
    ReLu(),
    MaxPool2D(pool_size=2, stride=2),
    Flatten(),
    Linear(64 * 7 * 7, 128),
    ReLu(),
    Dropout(rate=0.5),
    Linear(128, 10),
    Softmax()
])

model.compile(optimizer=Adam(learning_rate=0.001), loss=CrossEntropyLoss())
history = model.fit(X_train, y_train, epochs=10, batch_size=32, clip_grad_norm=1.0)
```

### RNN Model (Sequence Classification)

```python
from Models import RNNModel
from Layers import Embedding, LSTM, Linear
from Activation_classes import Softmax
from Losses import CrossEntropyLoss
from Optimizer import Adam

# Build RNN for text classification
model = RNNModel([
    Embedding(vocab_size=10000, embed_dim=128),
    LSTM(input_size=128, hidden_size=256, return_sequences=False),
    Linear(256, 64),
    ReLu(),
    Linear(64, num_classes),
    Softmax()
])

model.compile(optimizer=Adam(learning_rate=0.001), loss=CrossEntropyLoss())
history = model.fit(X_train, y_train, epochs=10, batch_size=32, clip_grad_norm=1.0)

# Generate sequences
generated = model.generate(start_tokens, max_length=100, temperature=0.8)
```

### Transformer Model (Language Modeling)

```python
from Models import Transformer
from Losses import CrossEntropyLoss
from Optimizer import AdamW

# Build GPT-style transformer
model = Transformer(
    vocab_size=50000,
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    max_seq_len=512,
    dropout=0.1,
    causal=True
)

model.compile(optimizer=AdamW(learning_rate=1e-4, weight_decay=0.01), loss=CrossEntropyLoss())
model.summary()

# Train on language modeling task
history = model.fit(X_train, y_train, epochs=10, batch_size=16, clip_grad_norm=1.0)

# Generate text
generated = model.generate(
    start_tokens=start_ids,
    max_length=100,
    temperature=0.8,
    top_k=40,
    top_p=0.9
)
```

### Using Gradient Clipping

```python
from utils import clip_grad_norm, clip_grad_value

# Clip during training
history = model.fit(X_train, y_train, epochs=10, clip_grad_norm=1.0)

# Or manually
output = model.forward(X_batch)
loss = loss_fn.forward(output, y_batch)
grad = loss_fn.backward()
model.backward(grad)

# Clip gradients before optimizer step
total_norm = clip_grad_norm(model.layers, max_norm=1.0)
clip_grad_value(model.layers, clip_value=0.5)

optimizer.step(model.layers)
```

### Using Regularization

```python
from utils import L1Regularization, L2Regularization, MaxNormConstraint

l2_reg = L2Regularization(lambda_reg=0.001)
max_norm = MaxNormConstraint(max_norm=3.0)

for epoch in range(epochs):
    # Forward and backward pass
    output = model.forward(X_batch)
    loss = loss_fn.forward(output, y_batch)
    
    # Add regularization loss
    reg_loss = l2_reg.loss(model.layers)
    total_loss = loss + reg_loss
    
    grad = loss_fn.backward()
    model.backward(grad)
    
    # Apply regularization gradients
    l2_reg.apply_gradients(model.layers)
    
    optimizer.step(model.layers)
    
    # Apply weight constraints
    max_norm.apply(model.layers)
```

### Using Callbacks and Schedulers

```python
from callbacks import EarlyStopping, ModelCheckpoint
from schedulers import ReduceLROnPlateau

early_stop = EarlyStopping(patience=5, mode='min')
checkpoint = ModelCheckpoint('best_model.npz', monitor='val_loss')
scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
```

### Model Saving and Loading

```python
model.save('model_weights.npz')

new_model = FNN(loss=CrossEntropyLoss(), optimizer=Adam())
new_model.load('model_weights.npz')
```

## Project Structure

```text
NexNet/
├── Activation_classes/     # Activation functions
├── Layers/                 # Neural network layers
├── Losses/                 # Loss functions
├── Models/                 # Model architectures (FNN, Sequential, CNN, RNN, Transformer)
├── Optimizer/              # Optimization algorithms
├── callbacks/              # Training callbacks
├── core/                   # Base classes (Module, Parameter)
├── data/                   # Data utilities
├── metrics/                # Evaluation metrics
├── schedulers/             # Learning rate schedulers
├── utils/                  # Utilities (initializers, regularization, grad_clip)
├── implementation/         # Example implementations
├── NLP/                    # NLP implementations
└── requirements.txt
```

## NLP Implementations

For NLP-related implementations (Word2Vec, GloVe, NER), see [Readme_NLP.md](https://github.com/chiruu12/NexNet/blob/main/NLP/Readme_NLP.md).

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add your feature"`
4. Push to branch: `git push origin feature/your-feature`
5. Open a Pull Request

## License

NexNet is licensed under the [MIT License](LICENSE).
