import numpy as np


class SGD:
    """
    Stochastic Gradient Descent optimizer.
    
    Updates parameters using the simple gradient descent rule:
    param = param - learning_rate * gradient
    """
    
    def __init__(self, learning_rate=0.01):
        """
        Initialize the SGD optimizer.
        
        Args:
            learning_rate: Learning rate for parameter updates.
        """
        self.learning_rate = learning_rate

    def step(self, layers):
        """
        Perform a single optimization step.
        
        Args:
            layers: List of layers with parameters to update.
        """
        for layer in layers:
            if hasattr(layer, 'W') and layer.dW is not None:
                layer.W -= self.learning_rate * layer.dW
                layer.b -= self.learning_rate * layer.db
            if hasattr(layer, 'gamma') and hasattr(layer, 'dgamma') and layer.dgamma is not None:
                layer.gamma -= self.learning_rate * layer.dgamma
                layer.beta -= self.learning_rate * layer.dbeta