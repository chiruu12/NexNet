import numpy as np 

class NAdam:
    # it is basically Adam but with nesterovs 
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7):
        """
        Initialize the NAdam optimizer.

        Args:
            learning_rate : Learning rate for the optimizer.
            beta1 : Exponential decay rate for the first moment estimate.
            beta2 : Exponential decay rate for the second moment estimate.
            epsilon : Small constant to prevent division by zero and ensure numerical stability.
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m_W = []  # First moment vector for weights
        self.m_b = []  # First moment vector for biases
        self.v_W = []  # Second moment vector for weights
        self.v_b = []  # Second moment vector for biases
        self.t = 0     # Time step

    def step(self, layers):
        """
        Perform a single optimization step with Adam.

        Args:
            layers: List of layers in the network. Each layer should have attributes `W` (weights), `b` (biases),
            `dW` (gradient of weights), and `db` (gradient of biases).
        """
        self.t += 1  # Increment time step
        
        for i, layer in enumerate(layers):
            if hasattr(layer, 'W'):
                # Initialize moment vectors if they don't exist
                if i > len(self.m_W):
                    self.m_W.append(np.zeros_like(layer.W))
                    self.m_b.append(np.zeros_like(layer.b))
                    self.v_W.append(np.zeros_like(layer.W))
                    self.v_b.append(np.zeros_like(layer.b))

                # Update first and second moment vectors
                self.m_W[i-1] = self.beta1 * self.m_W[i-1] + (1 - self.beta1) * layer.dW
                self.m_b[i-1] = self.beta1 * self.m_b[i-1] + (1 - self.beta1) * layer.db
                self.v_W[i-1] = self.beta2 * self.v_W[i-1] + (1 - self.beta2) * (layer.dW ** 2)
                self.v_b[i-1] = self.beta2 * self.v_b[i-1] + (1 - self.beta2) * (layer.db ** 2)

                # Bias correction
                m_W_hat = self.m_W[i-1] / (1 - self.beta1 ** self.t)
                m_b_hat = self.m_b[i-1] / (1 - self.beta1 ** self.t)
                v_W_hat = self.v_W[i-1] / (1 - self.beta2 ** self.t)
                v_b_hat = self.v_b[i-1] / (1 - self.beta2 ** self.t)

                # Update weights and biases with Nesterov correction 
                layer.W -= self.learning_rate * (self.beta1 * m_W_hat + (1 - self.beta1) * layer.dW) / (np.sqrt(v_W_hat) + self.epsilon)
                layer.b -= self.learning_rate * (self.beta1 * m_b_hat + (1 - self.beta1) * layer.db) / (np.sqrt(v_b_hat) + self.epsilon)