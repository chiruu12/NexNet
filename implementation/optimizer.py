import numpy as np
class Optimizer:
    def __init__(self, learning_rate=0.01, **kwargs):
        self.learning_rate = learning_rate

    def step(self, layers):
        """
        Perform a single optimization step. This method needs to be overridden
        by specific optimizer classes.
        
        Args:
            layers: List of layers in the network. Each layer should have attributes
                    `W` (weights), `b` (biases), `dW` (gradient of weights), and `db` (gradient of biases).
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        """
        Initialize the Stochastic Gradient Descent (SGD) optimizer.

        Args:
            learning_rate (float): Learning rate for the optimizer.
        """
        self.learning_rate = learning_rate

    def step(self, layers):
        """
        Perform a single optimization step by updating the weights and biases of the given layers.

        Args:
            layers  : List of layers in the network. Each layer should have attributes or if they dont we wont calculate for 
            `W` (weights), `b` (biases), `dW` (gradient of weights), and `db` (gradient of biases).

        This method iterates over each layer and updates its weights and biases using the computed gradients.
        """
        for layer in layers:
            if hasattr(layer, 'W'): # going to check if the layer is having a W attribute or we can say it is a linear layer or not!
                # Updates weights and biases for layers with weight and bias attributes
                layer.W -= self.learning_rate * layer.dW
                layer.b -= self.learning_rate * layer.db
                
                
class Momentum(Optimizer):
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
class AdaGrad(Optimizer):
    def __init__(self, learning_rate, epsilon=1e-8):
        """
        Initialize the AdaGrad optimizer.

        Args:
            learning_rate : Learning rate for the optimizer.
            epsilon : Small constant to prevent division by zero and ensure numerical stability.
        """
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.v_W = []  # Accumulated squared gradients for weights
        self.v_b = []  # Accumulated squared gradients for biases
    
    def step(self, layers):
        """
        Perform a single optimization step with AdaGrad.

        Args:
            layers: List of layers in the network. Each layer should have attributes `W` (weights), `b` (biases),
                    `dW` (gradient of weights), and `db` (gradient of biases).
        """
        for i, layer in enumerate(layers):
            if hasattr(layer, 'W'):
                # Initialize accumulated squared gradients if they don't exist
                if i >= len(self.v_W):
                    self.v_W.append(np.zeros_like(layer.W))  # Accumulated squared gradients for weights
                    self.v_b.append(np.zeros_like(layer.b))  # Accumulated squared gradients for biases

                # Update accumulated squared gradients
                self.v_W[i] += layer.dW ** 2
                self.v_b[i] += layer.db ** 2

                # Update weights and biases using accumulated squared gradients
                layer.W -= self.learning_rate * layer.dW / np.sqrt(self.v_W[i] + self.epsilon)
                layer.b -= self.learning_rate * layer.db / np.sqrt(self.v_b[i] + self.epsilon)


class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.01, Beta=0.9, epsilon=1e-7):
        """
        Initialize the RMSProp optimizer.

        Args:
            learning_rate : Learning rate for the optimizer.
            Beta : Decay factor for the running average of squared gradients.
            epsilon : Small constant to prevent division by zero and ensure numerical stability.
        """
        self.learning_rate = learning_rate
        self.Beta = Beta
        self.epsilon = epsilon
        self.v_W = []  # Running average of squared gradients for weights
        self.v_b = []  # Running average of squared gradients for biases
    
    def step(self, layers):
        """
        Perform a single optimization step with RMSProp.

        Args:
            layers: List of layers in the network. Each layer should have attributes `W` (weights), `b` (biases),
                    `dW` (gradient of weights), and `db` (gradient of biases).
        """
        for i, layer in enumerate(layers):
            if hasattr(layer, 'W'):
                # Initialize running averages if they don't exist
                if i >= len(self.v_W):
                    self.v_W.append(np.zeros_like(layer.W))  # Squared gradients for weights
                    self.v_b.append(np.zeros_like(layer.b))  # Squared gradients for biases

                # Update running average of squared gradients
                self.v_W[i] = self.Beta * self.v_W[i] + (1 - self.Beta) * layer.dW ** 2
                self.v_b[i] = self.Beta * self.v_b[i] + (1 - self.Beta) * layer.db ** 2

                # Update weights and biases using the running average of squared gradients
                layer.W -= self.learning_rate * layer.dW / np.sqrt(self.v_W[i] + self.epsilon)
                layer.b -= self.learning_rate * layer.db / np.sqrt(self.v_b[i] + self.epsilon)

class AdaDelta(Optimizer):
    def __init__(self, Beta, epsilon=1e-7):
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
        for i, layer in enumerate(layers):
            if hasattr(layer, 'W'):
                # Initialize running averages if they don't exist
                if i >= len(self.v_W):
                    self.v_W.append(np.zeros_like(layer.W))  # Gradient squared averages for weights
                    self.v_b.append(np.zeros_like(layer.b))  # Gradient squared averages for biases
                    self.u_W.append(np.zeros_like(layer.W))  # Update squared averages for weights
                    self.u_b.append(np.zeros_like(layer.b))  # Update squared averages for biases
                    
                # Update running averages of squared gradients
                self.v_W[i] = self.Beta * self.v_W[i] + (1 - self.Beta) * layer.dW ** 2
                self.v_b[i] = self.Beta * self.v_b[i] + (1 - self.Beta) * layer.db ** 2

                # Compute parameter updates
                delta_w = layer.dW * np.sqrt(self.u_W[i] + self.epsilon) / np.sqrt(self.v_W[i] + self.epsilon)
                delta_b = layer.db * np.sqrt(self.u_b[i] + self.epsilon) / np.sqrt(self.v_b[i] + self.epsilon)
                
                # Update running averages of squared updates
                self.u_W[i] = self.Beta * self.u_W[i] + (1 - self.Beta) * delta_w ** 2
                self.u_b[i] = self.Beta * self.u_b[i] + (1 - self.Beta) * delta_b ** 2
                
                # Update weights and biases
                layer.W -= delta_w
                layer.b -= delta_b
class Adam(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7):
        """
        Initialize the Adam optimizer.

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
                if i >= len(self.m_W):
                    self.m_W.append(np.zeros_like(layer.W))
                    self.m_b.append(np.zeros_like(layer.b))
                    self.v_W.append(np.zeros_like(layer.W))
                    self.v_b.append(np.zeros_like(layer.b))

                # Update first and second moment vectors
                self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * layer.dW
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.db
                self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (layer.dW ** 2)
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (layer.db ** 2)

                # Bias correction
                m_W_hat = self.m_W[i] / (1 - self.beta1 ** self.t)
                m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
                v_W_hat = self.v_W[i] / (1 - self.beta2 ** self.t)
                v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

                # Update weights and biases
                layer.W -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
                layer.b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
                
class NAdam(Optimizer):
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
                if i >= len(self.m_W):
                    self.m_W.append(np.zeros_like(layer.W))
                    self.m_b.append(np.zeros_like(layer.b))
                    self.v_W.append(np.zeros_like(layer.W))
                    self.v_b.append(np.zeros_like(layer.b))

                # Update first and second moment vectors
                self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * layer.dW
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.db
                self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (layer.dW ** 2)
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (layer.db ** 2)

                # Bias correction
                m_W_hat = self.m_W[i] / (1 - self.beta1 ** self.t)
                m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
                v_W_hat = self.v_W[i] / (1 - self.beta2 ** self.t)
                v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

                # Update weights and biases with Nesterov correction 
                layer.W -= self.learning_rate * (self.beta1 * m_W_hat + (1 - self.beta1) * layer.dW) / (np.sqrt(v_W_hat) + self.epsilon)
                layer.b -= self.learning_rate * (self.beta1 * m_b_hat + (1 - self.beta1) * layer.db) / (np.sqrt(v_b_hat) + self.epsilon)
