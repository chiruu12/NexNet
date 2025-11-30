from core.backend import get_array_module, sqrt, zeros_like


class AdaGrad:
    """
    AdaGrad optimizer.
    
    Adapts the learning rate for each parameter based on the historical
    sum of squared gradients. Parameters with larger gradients get smaller
    learning rates.
    """
    
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        """
        Initialize the AdaGrad optimizer.

        Args:
            learning_rate: Learning rate for the optimizer.
            epsilon: Small constant to prevent division by zero.
        """
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.v_W = []
        self.v_b = []
        self.v_gamma = []
        self.v_beta_bn = []

    def step(self, layers):
        """
        Perform a single optimization step with AdaGrad.

        Args:
            layers: List of layers in the network.
        """
        linear_idx = 0
        bn_idx = 0
        
        for layer in layers:
            if hasattr(layer, 'W') and hasattr(layer, 'dW'):
                if linear_idx >= len(self.v_W):
                    self.v_W.append(zeros_like(layer.W))
                    self.v_b.append(zeros_like(layer.b))

                self.v_W[linear_idx] += layer.dW ** 2
                self.v_b[linear_idx] += layer.db ** 2

                layer.W -= self.learning_rate * layer.dW / (sqrt(self.v_W[linear_idx]) + self.epsilon)
                layer.b -= self.learning_rate * layer.db / (sqrt(self.v_b[linear_idx]) + self.epsilon)
                linear_idx += 1
                
            elif hasattr(layer, 'gamma') and hasattr(layer, 'dgamma'):
                if bn_idx >= len(self.v_gamma):
                    self.v_gamma.append(zeros_like(layer.gamma))
                    self.v_beta_bn.append(zeros_like(layer.beta))

                self.v_gamma[bn_idx] += layer.dgamma ** 2
                self.v_beta_bn[bn_idx] += layer.dbeta ** 2

                layer.gamma -= self.learning_rate * layer.dgamma / (sqrt(self.v_gamma[bn_idx]) + self.epsilon)
                layer.beta -= self.learning_rate * layer.dbeta / (sqrt(self.v_beta_bn[bn_idx]) + self.epsilon)
                bn_idx += 1

