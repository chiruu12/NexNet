import numpy as np


class AdaDelta:
    """
    AdaDelta optimizer.
    
    Extension of AdaGrad that reduces its aggressive, monotonically decreasing
    learning rate. Uses exponentially decaying average of squared gradients.
    """
    
    def __init__(self, rho=0.95, epsilon=1e-7):
        """
        Initialize the AdaDelta optimizer.

        Args:
            rho: Decay factor for the running averages.
            epsilon: Small constant to prevent division by zero.
        """
        self.rho = rho
        self.epsilon = epsilon
        self.u_W = []
        self.u_b = []
        self.v_W = []
        self.v_b = []
        self.u_gamma = []
        self.u_beta_bn = []
        self.v_gamma = []
        self.v_beta_bn = []

    def step(self, layers):
        """
        Perform a single optimization step with AdaDelta.

        Args:
            layers: List of layers in the network.
        """
        linear_idx = 0
        bn_idx = 0
        
        for layer in layers:
            if hasattr(layer, 'W') and hasattr(layer, 'dW'):
                if linear_idx >= len(self.v_W):
                    self.v_W.append(np.zeros_like(layer.W))
                    self.v_b.append(np.zeros_like(layer.b))
                    self.u_W.append(np.zeros_like(layer.W))
                    self.u_b.append(np.zeros_like(layer.b))

                self.v_W[linear_idx] = self.rho * self.v_W[linear_idx] + (1 - self.rho) * layer.dW ** 2
                self.v_b[linear_idx] = self.rho * self.v_b[linear_idx] + (1 - self.rho) * layer.db ** 2

                delta_w = layer.dW * np.sqrt(self.u_W[linear_idx] + self.epsilon) / np.sqrt(self.v_W[linear_idx] + self.epsilon)
                delta_b = layer.db * np.sqrt(self.u_b[linear_idx] + self.epsilon) / np.sqrt(self.v_b[linear_idx] + self.epsilon)

                self.u_W[linear_idx] = self.rho * self.u_W[linear_idx] + (1 - self.rho) * delta_w ** 2
                self.u_b[linear_idx] = self.rho * self.u_b[linear_idx] + (1 - self.rho) * delta_b ** 2

                layer.W -= delta_w
                layer.b -= delta_b
                linear_idx += 1
                
            elif hasattr(layer, 'gamma') and hasattr(layer, 'dgamma'):
                if bn_idx >= len(self.v_gamma):
                    self.v_gamma.append(np.zeros_like(layer.gamma))
                    self.v_beta_bn.append(np.zeros_like(layer.beta))
                    self.u_gamma.append(np.zeros_like(layer.gamma))
                    self.u_beta_bn.append(np.zeros_like(layer.beta))

                self.v_gamma[bn_idx] = self.rho * self.v_gamma[bn_idx] + (1 - self.rho) * layer.dgamma ** 2
                self.v_beta_bn[bn_idx] = self.rho * self.v_beta_bn[bn_idx] + (1 - self.rho) * layer.dbeta ** 2

                delta_gamma = layer.dgamma * np.sqrt(self.u_gamma[bn_idx] + self.epsilon) / np.sqrt(self.v_gamma[bn_idx] + self.epsilon)
                delta_beta = layer.dbeta * np.sqrt(self.u_beta_bn[bn_idx] + self.epsilon) / np.sqrt(self.v_beta_bn[bn_idx] + self.epsilon)

                self.u_gamma[bn_idx] = self.rho * self.u_gamma[bn_idx] + (1 - self.rho) * delta_gamma ** 2
                self.u_beta_bn[bn_idx] = self.rho * self.u_beta_bn[bn_idx] + (1 - self.rho) * delta_beta ** 2

                layer.gamma -= delta_gamma
                layer.beta -= delta_beta
                bn_idx += 1