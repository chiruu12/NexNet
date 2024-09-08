---
### A neural network for MNIST data set using only numpy!!!

## Objective

In this we created a neural network framework from scratch using NumPy. The goal was to implement a framework that includes core components of a neural network such as layers, activation functions, loss functions, and an optimizer. Finally, we trained the neural network on the MNIST dataset and achieved at least 84% accuracy on the test dataset from Kaggle and the highest achieved being 93%.

Caution: Most of the comments in this code are AI-generated.I have reviewed the entire code,but if you find any typos or mistakes, please contact me.
## Components Implemented

The framework consists of the following key components:

1. **Linear Layer Class**:
   - Implements a fully connected layer.
   - Methods: `forward`, `backward`.

2. **ReLU Activation Class**:
   - Implements the ReLU activation function.
   - Methods: `forward`, `backward`.

3. **Sigmoid Activation Class**:
   - Implements the Sigmoid activation function.
   - Methods: `forward`, `backward`.

4. **Tanh Activation Class**:
   - Implements the Tanh activation function.
   - Methods: `forward`, `backward`.

5. **Softmax Activation Class**:
   - Implements the Softmax activation function.
   - Methods: `forward`, `backward`.

6. **Cross-Entropy Loss Class**:
   - Implements the cross-entropy loss function.
   - Methods: `forward`, `backward`.

7. **Mean Squared Error (MSE) Loss Class**:
   - Implements the MSE loss function.
   - Methods: `forward`, `backward`.

8. **SGD Optimizer Class**:
   - Implements the stochastic gradient descent optimizer.
   - Methods: `step`.
     
9. **OneHot Encoding Class (OneHot)**:
   - Provides utilities for one-hot encoding and decoding.
   - Methods: `convert_to_one_hot`,`one_hot_to_label`
     
10. **Model Class**:
    - Wraps all components into a cohesive model.
    - Methods: `add_layer`, `compile`, `train`, `predict`, `evaluate`, `save`, `load`.
   
## Installation

To use the framework, you need to have Python installed with NumPy,sklearn,pandas and sys. You can install NumPy using pip:

```bash
pip install numpy sklearn pandas sys
```
# OR
Install Required Packages:
```bash
pip install -r requirements.txt
```
## Usage Instructions

### 1. Import the Framework

To use the framework, import the necessary classes into your Kaggle notebook or Python script:

```python
from Linear import Linear
from Activation_classes import ReLU, Softmax
from losses import CrossEntropyLoss, MeanSquaredErrorLoss
from utils import SGD,one_hot
from Model import Model
```

### 2. Create and Train a Model

Here's an example of how to create and train a model using the framework:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
train = pd.read_csv('/path/to/train.csv')
test = pd.read_csv('/path/to/test.csv')
submission = pd.read_csv('/path/to/sample_submission.csv')

# Prepare data
X = train.drop(columns=['label']).to_numpy()
y = train['label'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Convert labels to one-hot encoding
one_hot_encoder = one_hot(num_classes=10)
y_train_one_hot = one_hot_encoder.convert_to_one_hot(y_train)
y_test_one_hot = one_hot_encoder.convert_to_one_hot(y_test)

# Initialize model
model = Model()
model.add_layer(Linear(input_dim=784, output_dim=128))
model.add_layer(ReLU())
model.add_layer(Linear(input_dim=128, output_dim=10))
model.add_layer(Softmax())

# Define loss function and optimizer
loss_function = CrossEntropyLoss()
optimizer = SGD(learning_rate=0.01)
model.compile(loss=loss_function, optimizer=optimizer)

# Train the model
model.train(X_train, y_train_one_hot, epochs=20, batch_size=64)

# Evaluate the model
predictions, test_loss, test_accuracy = model.evaluate(X_test, y_test_one_hot)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Predict on test data
test_array = model.predict(test.to_numpy())
y_predictions = one_hot_encoder.one_hot_to_label(test_array)

# Prepare and save submission
ans = pd.DataFrame({
    'ImageId': submission["ImageId"].to_numpy(),
    'Label': y_predictions
})
ans.to_csv('/path/to/submission.csv', index=False)
```

### 3. Save and Load Models

To save and load models:

```python
# Save model
model.save('/path')

# Load model
loaded_model = Model.load('/path')
```

## Kaggle Notebook

You can view and execute the Kaggle notebook that demonstrates the usage of this framework [here](https://www.kaggle.com/{your-kaggle-username}/{your-notebook-name}).

## Contributing

If you want to contribute to this project, please fork the repository and submit a pull request. 

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The MNIST dataset used for training and evaluation.
- Kaggle for providing the dataset and platform.

---
