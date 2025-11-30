from core.backend import get_array_module, zeros_like


class Momentum:
    """
    SGD with Momentum optimizer.
    
    Accelerates SGD in the relevant direction and dampens oscillations
    by accumulating a velocity vector in directions of persistent reduction.
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Initialize the Momentum optimizer.

        Args:
            learning_rate: Learning rate for the optimizer.
            momentum: Momentum factor (typically 0.9).
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v_W = []
        self.v_b = []
        self.v_gamma = []
        self.v_beta_bn = []

    def step(self, layers):
        """
        Perform a single optimization step with momentum.

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

                self.v_W[linear_idx] = self.momentum * self.v_W[linear_idx] - self.learning_rate * layer.dW
                self.v_b[linear_idx] = self.momentum * self.v_b[linear_idx] - self.learning_rate * layer.db

                layer.W += self.v_W[linear_idx]
                layer.b += self.v_b[linear_idx]
                linear_idx += 1
                
            elif hasattr(layer, 'gamma') and hasattr(layer, 'dgamma'):
                if bn_idx >= len(self.v_gamma):
                    self.v_gamma.append(zeros_like(layer.gamma))
                    self.v_beta_bn.append(zeros_like(layer.beta))

                self.v_gamma[bn_idx] = self.momentum * self.v_gamma[bn_idx] - self.learning_rate * layer.dgamma
                self.v_beta_bn[bn_idx] = self.momentum * self.v_beta_bn[bn_idx] - self.learning_rate * layer.dbeta

                layer.gamma += self.v_gamma[bn_idx]
                layer.beta += self.v_beta_bn[bn_idx]
                bn_idx += 1