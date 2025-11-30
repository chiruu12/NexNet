from core.backend import get_array_module, sqrt, zeros_like


class NAdam:
    """
    NAdam optimizer (Adam with Nesterov momentum).
    
    Combines Adam's adaptive learning rates with Nesterov's
    lookahead momentum for potentially faster convergence.
    """
    
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7):
        """
        Initialize the NAdam optimizer.

        Args:
            learning_rate: Learning rate for the optimizer.
            beta1: Exponential decay rate for the first moment estimate.
            beta2: Exponential decay rate for the second moment estimate.
            epsilon: Small constant to prevent division by zero.
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_W = []
        self.m_b = []
        self.v_W = []
        self.v_b = []
        self.m_gamma = []
        self.m_beta_bn = []
        self.v_gamma = []
        self.v_beta_bn = []
        self.t = 0

    def step(self, layers):
        """
        Perform a single optimization step with NAdam.

        Args:
            layers: List of layers in the network.
        """
        self.t += 1
        linear_idx = 0
        bn_idx = 0
        
        for layer in layers:
            if hasattr(layer, 'W'):
                if linear_idx >= len(self.m_W):
                    self.m_W.append(zeros_like(layer.W))
                    self.m_b.append(zeros_like(layer.b))
                    self.v_W.append(zeros_like(layer.W))
                    self.v_b.append(zeros_like(layer.b))

                self.m_W[linear_idx] = self.beta1 * self.m_W[linear_idx] + (1 - self.beta1) * layer.dW
                self.m_b[linear_idx] = self.beta1 * self.m_b[linear_idx] + (1 - self.beta1) * layer.db
                self.v_W[linear_idx] = self.beta2 * self.v_W[linear_idx] + (1 - self.beta2) * (layer.dW ** 2)
                self.v_b[linear_idx] = self.beta2 * self.v_b[linear_idx] + (1 - self.beta2) * (layer.db ** 2)

                m_W_hat = self.m_W[linear_idx] / (1 - self.beta1 ** self.t)
                m_b_hat = self.m_b[linear_idx] / (1 - self.beta1 ** self.t)
                v_W_hat = self.v_W[linear_idx] / (1 - self.beta2 ** self.t)
                v_b_hat = self.v_b[linear_idx] / (1 - self.beta2 ** self.t)

                layer.W -= self.learning_rate * (self.beta1 * m_W_hat + (1 - self.beta1) * layer.dW / (1 - self.beta1 ** self.t)) / (sqrt(v_W_hat) + self.epsilon)
                layer.b -= self.learning_rate * (self.beta1 * m_b_hat + (1 - self.beta1) * layer.db / (1 - self.beta1 ** self.t)) / (sqrt(v_b_hat) + self.epsilon)
                linear_idx += 1
                
            elif hasattr(layer, 'gamma') and hasattr(layer, 'dgamma'):
                if bn_idx >= len(self.m_gamma):
                    self.m_gamma.append(zeros_like(layer.gamma))
                    self.m_beta_bn.append(zeros_like(layer.beta))
                    self.v_gamma.append(zeros_like(layer.gamma))
                    self.v_beta_bn.append(zeros_like(layer.beta))

                self.m_gamma[bn_idx] = self.beta1 * self.m_gamma[bn_idx] + (1 - self.beta1) * layer.dgamma
                self.m_beta_bn[bn_idx] = self.beta1 * self.m_beta_bn[bn_idx] + (1 - self.beta1) * layer.dbeta
                self.v_gamma[bn_idx] = self.beta2 * self.v_gamma[bn_idx] + (1 - self.beta2) * (layer.dgamma ** 2)
                self.v_beta_bn[bn_idx] = self.beta2 * self.v_beta_bn[bn_idx] + (1 - self.beta2) * (layer.dbeta ** 2)

                m_gamma_hat = self.m_gamma[bn_idx] / (1 - self.beta1 ** self.t)
                m_beta_hat = self.m_beta_bn[bn_idx] / (1 - self.beta1 ** self.t)
                v_gamma_hat = self.v_gamma[bn_idx] / (1 - self.beta2 ** self.t)
                v_beta_hat = self.v_beta_bn[bn_idx] / (1 - self.beta2 ** self.t)

                layer.gamma -= self.learning_rate * (self.beta1 * m_gamma_hat + (1 - self.beta1) * layer.dgamma / (1 - self.beta1 ** self.t)) / (sqrt(v_gamma_hat) + self.epsilon)
                layer.beta -= self.learning_rate * (self.beta1 * m_beta_hat + (1 - self.beta1) * layer.dbeta / (1 - self.beta1 ** self.t)) / (sqrt(v_beta_hat) + self.epsilon)
                bn_idx += 1