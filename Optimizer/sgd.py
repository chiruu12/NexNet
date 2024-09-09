import numpy as np
class SGD:
    def __init__(self, learning_rate=0.01):
        """
        Initialize the Stochastic Gradient Descent (SGD) optimizer.

        Args:
            learning_rate (float): Learning rate for the optimizer.
        """
        self.learning_rate = learning_rate

    def step(self, layers):
        """
        Perform a single optimization step by updating the weights and biases of the given layers.

        Args:
            layers  : List of layers in the network. Each layer should have attributes or if they dont we wont calculate for 
            `W` (weights), `b` (biases), `dW` (gradient of weights), and `db` (gradient of biases).

        This method iterates over each layer and updates its weights and biases using the computed gradients.
        """
        for layer in layers:
            if hasattr(layer, 'W'): # going to check if the layer is having a W attribute or we can say it is a linear layer or not!
                # Updates weights and biases for layers with weight and bias attributes
                layer.W -= self.learning_rate * layer.dW
                layer.b -= self.learning_rate * layer.db