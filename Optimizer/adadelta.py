import numpy as np

class AdaDelta:
    def __init__(self, Beta=0.945, epsilon=1e-7):
        """
        Initialize the AdaDelta optimizer.

        Args:
            Beta : Decay factor for the running averages.
            epsilon : Small constant to prevent division by zero and ensure numerical stability.
        """
        self.Beta = Beta
        self.epsilon = epsilon
        self.u_W = []  # Running average of squared updates for weights
        self.u_b = []  # Running average of squared updates for biases
        self.v_W = []  # Running average of squared gradients for weights
        self.v_b = []  # Running average of squared gradients for biases
    
    def step(self, layers):
        """
        Perform a single optimization step with AdaDelta.

        Args:
            layers: List of layers in the network. Each layer should have attributes `W` (weights), `b` (biases),
            `dW` (gradient of weights), and `db` (gradient of biases).
        """
        i=0
        for layer in layers:
            if hasattr(layer, 'W'):
                i+=1
                # Initialize running averages if they don't exist
                if i > len(self.v_W):
                    self.v_W.append(np.zeros_like(layer.W))  # Gradient squared averages for weights
                    self.v_b.append(np.zeros_like(layer.b))  # Gradient squared averages for biases
                    self.u_W.append(np.zeros_like(layer.W))  # Update squared averages for weights
                    self.u_b.append(np.zeros_like(layer.b))  # Update squared averages for biases
                    
                # Update running averages of squared gradients
                self.v_W[i-1] = self.Beta * self.v_W[i-1] + (1 - self.Beta) * layer.dW ** 2
                self.v_b[i-1] = self.Beta * self.v_b[i-1] + (1 - self.Beta) * layer.db ** 2

                # Compute parameter updates
                delta_w = layer.dW * np.sqrt(self.u_W[i-1] + self.epsilon) / np.sqrt(self.v_W[i-1] + self.epsilon)
                delta_b = layer.db * np.sqrt(self.u_b[i-1] + self.epsilon) / np.sqrt(self.v_b[i-1] + self.epsilon)
                
                # Update running averages of squared updates
                self.u_W[i-1] = self.Beta * self.u_W[i-1] + (1 - self.Beta) * delta_w ** 2
                self.u_b[i-1] = self.Beta * self.u_b[i-1] + (1 - self.Beta) * delta_b ** 2
                
                # Update weights and biases
                layer.W -= delta_w
                layer.b -= delta_b