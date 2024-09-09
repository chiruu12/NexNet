import numpy as np
class AdaGrad:
    def __init__(self, learning_rate, epsilon=1e-8):
        """
        Initialize the AdaGrad optimizer.

        Args:
            learning_rate : Learning rate for the optimizer.
            epsilon : Small constant to prevent division by zero and ensure numerical stability.
        """
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.v_W = []  # Accumulated squared gradients for weights
        self.v_b = []  # Accumulated squared gradients for biases
    
    def step(self, layers):
        """
        Perform a single optimization step with AdaGrad.

        Args:
            layers: List of layers in the network. Each layer should have attributes `W` (weights), `b` (biases),
                    `dW` (gradient of weights), and `db` (gradient of biases).
        """
        for i, layer in enumerate(layers):
            if hasattr(layer, 'W'):
                # Initialize accumulated squared gradients if they don't exist
                if i >= len(self.v_W):
                    self.v_W.append(np.zeros_like(layer.W))  # Accumulated squared gradients for weights
                    self.v_b.append(np.zeros_like(layer.b))  # Accumulated squared gradients for biases

                # Update accumulated squared gradients
                self.v_W[i] += layer.dW ** 2
                self.v_b[i] += layer.db ** 2

                # Update weights and biases using accumulated squared gradients
                layer.W -= self.learning_rate * layer.dW / np.sqrt(self.v_W[i] + self.epsilon)
                layer.b -= self.learning_rate * layer.db / np.sqrt(self.v_b[i] + self.epsilon)

