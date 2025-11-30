import numpy as np


class RMSProp:
    """
    RMSProp optimizer.
    
    Maintains a moving average of squared gradients to normalize
    the gradient, preventing the learning rate from becoming too
    large or too small.
    """
    
    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-7):
        """
        Initialize the RMSProp optimizer.

        Args:
            learning_rate: Learning rate for the optimizer.
            beta: Decay factor for the running average of squared gradients.
            epsilon: Small constant to prevent division by zero.
        """
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v_W = []
        self.v_b = []
        self.v_gamma = []
        self.v_beta_bn = []

    def step(self, layers):
        """
        Perform a single optimization step with RMSProp.

        Args:
            layers: List of layers in the network.
        """
        linear_idx = 0
        bn_idx = 0
        
        for layer in layers:
            if hasattr(layer, 'W'):
                if linear_idx >= len(self.v_W):
                    self.v_W.append(np.zeros_like(layer.W))
                    self.v_b.append(np.zeros_like(layer.b))

                self.v_W[linear_idx] = self.beta * self.v_W[linear_idx] + (1 - self.beta) * layer.dW ** 2
                self.v_b[linear_idx] = self.beta * self.v_b[linear_idx] + (1 - self.beta) * layer.db ** 2

                layer.W -= self.learning_rate * layer.dW / (np.sqrt(self.v_W[linear_idx]) + self.epsilon)
                layer.b -= self.learning_rate * layer.db / (np.sqrt(self.v_b[linear_idx]) + self.epsilon)
                linear_idx += 1
                
            elif hasattr(layer, 'gamma') and hasattr(layer, 'dgamma'):
                if bn_idx >= len(self.v_gamma):
                    self.v_gamma.append(np.zeros_like(layer.gamma))
                    self.v_beta_bn.append(np.zeros_like(layer.beta))

                self.v_gamma[bn_idx] = self.beta * self.v_gamma[bn_idx] + (1 - self.beta) * layer.dgamma ** 2
                self.v_beta_bn[bn_idx] = self.beta * self.v_beta_bn[bn_idx] + (1 - self.beta) * layer.dbeta ** 2

                layer.gamma -= self.learning_rate * layer.dgamma / (np.sqrt(self.v_gamma[bn_idx]) + self.epsilon)
                layer.beta -= self.learning_rate * layer.dbeta / (np.sqrt(self.v_beta_bn[bn_idx]) + self.epsilon)
                bn_idx += 1