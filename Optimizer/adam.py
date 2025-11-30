from core.backend import get_array_module, sqrt, zeros_like


class Adam:
    """
    Adam (Adaptive Moment Estimation) optimizer.
    
    Combines the benefits of AdaGrad and RMSProp with momentum,
    using adaptive learning rates for each parameter.
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize the Adam optimizer.
        
        Args:
            learning_rate: Learning rate for parameter updates.
            beta1: Exponential decay rate for first moment estimates.
            beta2: Exponential decay rate for second moment estimates.
            epsilon: Small constant for numerical stability.
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m_W = []
        self.m_b = []
        self.v_W = []
        self.v_b = []
        self.t = 0

    def step(self, layers):
        """
        Perform a single optimization step.
        
        Args:
            layers: List of layers with parameters to update.
        """
        self.t += 1
        layer_idx = 0
        
        for layer in layers:
            if hasattr(layer, 'W') and hasattr(layer, 'dW') and layer.dW is not None:
                if layer_idx >= len(self.m_W):
                    self.m_W.append(zeros_like(layer.W))
                    self.v_W.append(zeros_like(layer.W))
                    if hasattr(layer, 'b') and layer.b is not None:
                        self.m_b.append(zeros_like(layer.b))
                        self.v_b.append(zeros_like(layer.b))
                    else:
                        self.m_b.append(None)
                        self.v_b.append(None)

                self.m_W[layer_idx] = self.beta1 * self.m_W[layer_idx] + (1 - self.beta1) * layer.dW
                self.v_W[layer_idx] = self.beta2 * self.v_W[layer_idx] + (1 - self.beta2) * (layer.dW ** 2)

                m_W_hat = self.m_W[layer_idx] / (1 - self.beta1 ** self.t)
                v_W_hat = self.v_W[layer_idx] / (1 - self.beta2 ** self.t)

                layer.W -= self.learning_rate * m_W_hat / (sqrt(v_W_hat) + self.epsilon)
                
                if hasattr(layer, 'b') and layer.b is not None and hasattr(layer, 'db') and layer.db is not None:
                    self.m_b[layer_idx] = self.beta1 * self.m_b[layer_idx] + (1 - self.beta1) * layer.db
                    self.v_b[layer_idx] = self.beta2 * self.v_b[layer_idx] + (1 - self.beta2) * (layer.db ** 2)

                    m_b_hat = self.m_b[layer_idx] / (1 - self.beta1 ** self.t)
                    v_b_hat = self.v_b[layer_idx] / (1 - self.beta2 ** self.t)

                    layer.b -= self.learning_rate * m_b_hat / (sqrt(v_b_hat) + self.epsilon)
                
                layer_idx += 1
