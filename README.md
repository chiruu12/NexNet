# NexNet ⚙️

NexNet is a neural network framework implemented from scratch using NumPy. It provides functionalities similar to PyTorch and TensorFlow, including various activation functions, loss functions, optimizers, and more. This README will guide you through the setup, usage, and features of the framework.

## Features

- **Activation Functions:**
  - `ReLU`: Rectified Linear Unit, introduces non-linearity by zeroing out negative values.
  - `Softmax`: Converts logits to probabilities, commonly used in the output layer for classification tasks.
  - `PReLU`: Parametric ReLU, allows for a learnable slope for negative values.
  - `Sigmoid`: Maps values to a range between 0 and 1, often used in binary classification.
  - `Tanh`: Maps values to a range between -1 and 1, helping with centering data.
  - `LeakyReLU`: Similar to ReLU but allows a small gradient when inputs are negative.
  - `ELU`: Exponential Linear Unit, helps speed up learning by smoothing the activation function.
  - `Swish`: Smooth, non-monotonic activation function that can improve model performance.
  - `Softplus`: A smooth approximation to ReLU, improving gradient flow.

- **Loss Functions:**
  - `CrossEntropyLoss`: Measures the performance of a classification model whose output is a probability value between 0 and 1.
  - `MeanSquaredErrorLoss`: Calculates the average of the squares of the errors between predicted and actual values.
  - `BinaryCrossEntropyLoss`: Measures the performance of a binary classification model.
  - `HuberLoss`: Combines the advantages of Mean Squared Error and Mean Absolute Error.
  - `PoissonLoss`: Used for count-based prediction tasks, measures the difference between predicted and actual counts.
  - `MeanAbsoluteErrorLoss`: Calculates the average of the absolute errors between predicted and actual values.
  - `CosineSimilarityLoss`: Measures the cosine of the angle between two vectors to determine their similarity.

- **Optimizers:**
  - `SGD`: Stochastic Gradient Descent, updates weights based on a subset of the data.
  - `Momentum`: Enhances SGD by considering past gradients to accelerate convergence.
  - `AdaGrad`: Adapts learning rates based on the frequency of updates for each parameter.
  - `Adam`: Combines the benefits of AdaGrad and RMSProp, including adaptive learning rates.
  - `NAdam`: Adam with Nesterov accelerated gradient, improving convergence speed.
  - `RMSProp`: Adapts learning rates based on recent gradients to maintain a moving average of the squared gradients.
  - `AdaDelta`: An extension of AdaGrad that reduces its aggressive, monotonically decreasing learning rate.

- **Initializers:**
  - `Xavier`: Initializes weights to maintain the variance of activations, useful for sigmoid and tanh activations.
  - `He`: Initializes weights to avoid issues with dying neurons in ReLU-based networks.
  - `Random`: Simple random initialization for small networks.
  - `Zero`: Initializes weights to zero (not typically recommended for deep networks).

- **Layers:**
  - `Linear`: Fully connected layer that performs a linear transformation.

## Installation

To use NexNet, clone the repository and change into the directory:

```bash
git clone https://github.com/chiruu12/NexNet.git
import os
os.chdir('NexNet')
```

## Usage

### Importing Modules

To get started with NexNet, you need to import the necessary modules:

```python
from Models import FNN, CNN
from Losses import CrossEntropyLoss, MeanSquaredErrorLoss, BinaryCrossEntropyLoss, HuberLoss, PoissonLoss, MeanAbsoluteErrorLoss, CosineSimilarityLoss
from Layers import Linear
from Activation_classes import ReLu, Softmax, PReLU, Sigmoid, Tanh, LeakyReLU, ELU, Swish, Softplus
from utils import one_hot, Initializer
from Optimizer import SGD, Momentum, AdaGrad, Adam, NAdam, RMSProp, AdaDelta
```

## Data Preparation

Prepare your dataset by splitting it into training and testing sets and one-hot encoding the labels:

```python
from sklearn.model_selection import train_test_split

# Example data split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

# One-hot encode the labels
num_classes = 10  # For MNIST
one_hot_encoder = one_hot(num_classes)
y_train_one_hot = one_hot_encoder.convert_to_one_hot(y_train)
y_test_one_hot = one_hot_encoder.convert_to_one_hot(y_test)
```

## Creating and Training a Model

Create and train a model using the `FNN` class:

```python
# Create the model
input_dim = X_train.shape[1]
model = FNN(optimizer=AdaDelta(), loss=CrossEntropyLoss())

# Add layers to the model
model.add_layer(Linear(input_dim=input_dim, output_dim=128, activation=PReLU()))
model.add_layer(Linear(input_dim=128, output_dim=128, activation=PReLU()))
model.add_layer(Linear(input_dim=128, output_dim=32, activation=PReLU()))
model.add_layer(Linear(32, num_classes))

# Train the model
model.train(X_train, y_train_one_hot, epochs=30, batch_size=64)

# Evaluate on test set
accuracy = model.evaluate(X_test, y_test_one_hot)
```


## Model Saving and Loading

To save and load model weights, NexNet provides functionality to persist and restore model states. This is useful for checkpointing and resuming training or for deploying models.

### Saving the Model

After training your model, you can save its weights to a file using the `save` method. The weights are saved in a `.npz` file format.

```python
# Save the model weights to a file
model.save('model_weights.npz')
```


This will create a file named model_weights.npz containing the weights and biases of all layers in the model.

### Loading the Model
To load a previously saved model, use the load method. This will restore the weights and biases from the file into your model.

```python
# Load the model weights from a file
model.load('model_weights.npz')
```
Make sure the model architecture matches the one used when the weights were saved. The load method will update the weights and biases of the layers according to the saved state.


## Future Enhancements

NexNet is an evolving project with a focus on building a robust and flexible neural network library. Here are some planned enhancements and future improvements:

1. **Expanded Layer Support**
   - Additional Layers: Integrate more advanced types of layers such as dropout, batch normalization, and attention mechanisms.
   - Custom Layer Support: Allow users to define and implement their own custom layers.

2. **Optimizer Enhancements**
   - Advanced Optimizers: Introduce more optimization algorithms such as RMSProp, AdamW, and L-BFGS.
   - Hyperparameter Tuning: Implement automatic hyperparameter tuning capabilities for optimizers.

3. **Enhanced Model Management**
   - Checkpoint: Add functionality to save and restore model checkpoints during training.
   - Model Serialization: Improve model saving and loading to support various formats and metadata.

4. **Visualization and Monitoring**
   - Training Progress: Incorporate tools for visualizing training progress, including loss and accuracy curves.
   - Model Inspection: Develop utilities for inspecting and analyzing model parameters and performance.

5. **Expanded Loss Functions**
   - Additional Losses: Include more loss functions such as Hinge Loss, Triplet Loss, and custom user-defined losses.
   - Advanced Metrics: Implement advanced evaluation metrics and performance measures.

6. **Improved Documentation**
   - Detailed Examples: Provide more comprehensive examples and tutorials for using different features and functionalities.
   - API Documentation: Enhance the API documentation for easier navigation and understanding.

7. **Performance Optimization**
   - Efficiency Improvements: Optimize the performance of core components to handle larger datasets and more complex models.
   - Parallel Computing: Explore options for parallel computing to accelerate training and inference.

   ## Contributions

Contributions to NexNet are welcome! If you have suggestions, improvements, or bug fixes, please follow these steps:

1. **Fork the Repository**
   - Create a fork of the repository on GitHub to make your changes.

2. **Clone Your Fork**
   - Clone your fork to your local machine using:
     ```bash
     git clone https://github.com/your-username/NexNet.git
     ```

3. **Create a Branch**
   - Create a new branch for your changes:
     ```bash
     git checkout -b feature/your-feature-name
     ```

4. **Make Changes**
   - Implement your changes or add new features.

5. **Commit and Push**
   - Commit your changes with a descriptive message:
     ```bash
     git add .
     git commit -m "Add detailed description of changes"
     ```
   - Push your branch to your fork:
     ```bash
     git push origin feature/your-feature-name
     ```

6. **Create a Pull Request**
   - Open a pull request on the original repository to propose your changes.

7. **Review and Feedback**
   - The project maintainers will review your pull request and provide feedback if necessary.

Thank you for contributing to NexNet!

## License

NexNet is licensed under the [MIT License](LICENSE). See the [LICENSE](LICENSE) file for more details.
