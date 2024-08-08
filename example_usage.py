import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys

# Import custom classes and functions
from losses import CrossEntropyLoss
from Activation_classes import ReLU, Softmax
from Linear import Linear
from Model import Model
from utils import one_hot, SGD

# Add dataset path to system path (adjust to actual dataset path)
sys.path.append('/kaggle/input/{your-dataset-name}')

# Load the dataset
#train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
#test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
#submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

# Prepare the data
#y_train = train["label"].to_numpy()
#X_train = train.drop(columns=['label']).to_numpy()
#X_test = test.to_numpy()  # Test data has no labels
#X_train, X_test, y_train, y_test = train_test_split(
#    X_train, y_train, test_size=0.20, random_state=0)

# Initialize the model
model = Model()
model.add_layer(Linear(input_dim=784, output_dim=128))
model.add_layer(ReLU())
model.add_layer(Linear(input_dim=128, output_dim=10))
model.add_layer(Softmax())

# Define loss function and optimizer
loss_function = CrossEntropyLoss()
optimizer = SGD(learning_rate=0.01)
model.compile(loss=loss_function, optimizer=optimizer)

# Convert labels to one-hot encoding
one_hot_encoder = one_hot(num_classes=10)
y_train_one_hot = one_hot_encoder.convert_to_one_hot(y_train)
y_test_one_hot = one_hot_encoder.convert_to_one_hot(y_test)

# Train the model
model.train(X_train, y_train_one_hot, epochs=20, batch_size=64)

# Evaluate the model
test_array, test_loss, test_accuracy = model.evaluate(X_test, y_test_one_hot)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Retrain on the same data (this is not typically recommended)
model.train(X_test, y_test_one_hot, epochs=20, batch_size=42)
test_array, test_loss, test_accuracy = model.evaluate(X_test, y_test_one_hot)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Predict on the test data
predictions = model.predict(X_test)
y_predictions = one_hot_encoder.one_hot_to_label(predictions)

# Prepare the submission
ans = pd.DataFrame({
    'ImageId': submission["ImageId"].to_numpy(),
    'Label': y_predictions
})

# Save the results to a CSV file
ans.to_csv('{your file path}', index=False)
