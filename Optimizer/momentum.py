import numpy as np
class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Initialize the Momentum optimizer.

        Args:
            learning_rate : Learning rate for the optimizer.
            momentum : Momentum factor.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v_W = []
        self.v_b = []

    def step(self, layers):
        """
        Perform a single optimization step with momentum by updating the weights and biases of the given layers.

        Args:
            layers  : List of layers in the network. Each layer should have attributes or if they dont we wont calculate for 
            `W` (weights), `b` (biases),  `dW` (gradient of weights), and `db` (gradient of biases).
        """
        for i, layer in enumerate(layers):
            if hasattr(layer, 'W'):
                # Initialize velocity if it doesn't exist
                #basically the first iteration 
                if i <= len(self.v_W):
                    self.v_W.append(np.zeros_like(layer.W))
                    self.v_b.append(np.zeros_like(layer.b))

                # Update velocity for weights and biases
                self.v_W[i] = self.momentum *self.v_W[i] - self.learning_rate *layer.dW
                self.v_b[i] = self.momentum *self.v_b[i] - self.learning_rate *layer.db

                # Update weights and biases using the velocity
                layer.W += self.v_W[i]
                layer.b += self.v_b[i]