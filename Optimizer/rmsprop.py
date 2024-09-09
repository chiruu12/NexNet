import numpy as np 
class RMSProp:
    def __init__(self, learning_rate=0.01, Beta=0.9, epsilon=1e-7):
        """
        Initialize the RMSProp optimizer.

        Args:
            learning_rate : Learning rate for the optimizer.
            Beta : Decay factor for the running average of squared gradients.
            epsilon : Small constant to prevent division by zero and ensure numerical stability.
        """
        self.learning_rate = learning_rate
        self.Beta = Beta
        self.epsilon = epsilon
        self.v_W = []  # Running average of squared gradients for weights
        self.v_b = []  # Running average of squared gradients for biases
    
    def step(self, layers):
        """
        Perform a single optimization step with RMSProp.

        Args:
            layers: List of layers in the network. Each layer should have attributes `W` (weights), `b` (biases),
                    `dW` (gradient of weights), and `db` (gradient of biases).
        """
        for i, layer in enumerate(layers):
            if hasattr(layer, 'W'):
                # Initialize running averages if they don't exist
                if i >= len(self.v_W):
                    self.v_W.append(np.zeros_like(layer.W))  # Squared gradients for weights
                    self.v_b.append(np.zeros_like(layer.b))  # Squared gradients for biases

                # Update running average of squared gradients
                self.v_W[i] = self.Beta * self.v_W[i] + (1 - self.Beta) * layer.dW ** 2
                self.v_b[i] = self.Beta * self.v_b[i] + (1 - self.Beta) * layer.db ** 2

                # Update weights and biases using the running average of squared gradients
                layer.W -= self.learning_rate * layer.dW / np.sqrt(self.v_W[i] + self.epsilon)
                layer.b -= self.learning_rate * layer.db / np.sqrt(self.v_b[i] + self.epsilon)