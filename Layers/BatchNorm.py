import numpy as np


class BatchNorm:
    """
    Batch Normalization layer for normalizing layer inputs.
    
    Normalizes the input by subtracting the batch mean and dividing by
    the batch standard deviation. Includes learnable scale (gamma) and
    shift (beta) parameters.
    """
    
    def __init__(self, num_features, epsilon=1e-5, momentum=0.1):
        """
        Initialize the Batch Normalization layer.
        
        Args:
            num_features: Number of features/channels in the input.
            epsilon: Small constant for numerical stability.
            momentum: Momentum for running mean and variance updates.
        """
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
        self.dgamma = None
        self.dbeta = None
        
        self.training = True
        
        self.input_normalized = None
        self.input_centered = None
        self.std = None
        self.batch_mean = None
        self.batch_var = None

    def forward(self, inputs):
        """
        Perform the forward pass of the Batch Normalization layer.
        
        Args:
            inputs: Input data of shape (batch_size, num_features).
        
        Returns:
            Normalized and scaled output of the same shape.
        """
        if self.training:
            self.batch_mean = np.mean(inputs, axis=0, keepdims=True)
            self.batch_var = np.var(inputs, axis=0, keepdims=True)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.batch_var
            
            self.input_centered = inputs - self.batch_mean
            self.std = np.sqrt(self.batch_var + self.epsilon)
            self.input_normalized = self.input_centered / self.std
        else:
            self.input_normalized = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        output = self.gamma * self.input_normalized + self.beta
        return output

    def backward(self, gradient_output):
        """
        Perform the backward pass of the Batch Normalization layer.
        
        Args:
            gradient_output: Gradient of the loss with respect to the output.
        
        Returns:
            Gradient of the loss with respect to the input.
        """
        batch_size = gradient_output.shape[0]
        
        self.dgamma = np.sum(gradient_output * self.input_normalized, axis=0, keepdims=True)
        self.dbeta = np.sum(gradient_output, axis=0, keepdims=True)
        
        dnormalized = gradient_output * self.gamma
        
        dvar = np.sum(dnormalized * self.input_centered * -0.5 * (self.batch_var + self.epsilon) ** (-1.5), axis=0, keepdims=True)
        
        dmean = np.sum(dnormalized * -1 / self.std, axis=0, keepdims=True) + dvar * np.mean(-2 * self.input_centered, axis=0, keepdims=True)
        
        dinputs = dnormalized / self.std + dvar * 2 * self.input_centered / batch_size + dmean / batch_size
        
        return dinputs

    def train(self):
        """Set the layer to training mode."""
        self.training = True

    def eval(self):
        """Set the layer to evaluation mode."""
        self.training = False
