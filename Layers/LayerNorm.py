import numpy as np


class LayerNorm:
    """
    Layer Normalization.
    
    Normalizes across the feature dimension (last axis) for each sample independently.
    Unlike BatchNorm, LayerNorm doesn't depend on batch statistics, making it
    suitable for sequence models and transformers.
    
    For input of shape (batch, seq_len, features), normalizes over the features dimension.
    """
    
    def __init__(self, normalized_shape, epsilon=1e-5):
        """
        Initialize Layer Normalization.
        
        Args:
            normalized_shape: The shape of the features to normalize (typically the last dimension).
            epsilon: Small constant for numerical stability.
        """
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.epsilon = epsilon
        
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
        
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        
    def forward(self, x):
        """
        Forward pass of Layer Normalization.
        
        Args:
            x: Input tensor. Normalization is applied over the last len(normalized_shape) dimensions.
            
        Returns:
            Normalized output with same shape as input.
        """
        self.input = x
        self.input_shape = x.shape
        
        axes = tuple(range(-len(self.normalized_shape), 0))
        
        self.mean = np.mean(x, axis=axes, keepdims=True)
        self.var = np.var(x, axis=axes, keepdims=True)
        
        self.x_centered = x - self.mean
        self.std = np.sqrt(self.var + self.epsilon)
        self.x_norm = self.x_centered / self.std
        
        self.output = self.gamma * self.x_norm + self.beta
        
        return self.output
    
    def backward(self, gradient_output):
        """
        Backward pass of Layer Normalization.
        
        Args:
            gradient_output: Gradient from the next layer.
            
        Returns:
            Gradient with respect to input.
        """
        axes = tuple(range(-len(self.normalized_shape), 0))
        n = np.prod([self.input_shape[i] for i in range(-len(self.normalized_shape), 0)])
        
        sum_axes = tuple(range(len(self.input_shape) - len(self.normalized_shape)))
        self.dgamma = np.sum(gradient_output * self.x_norm, axis=sum_axes)
        self.dbeta = np.sum(gradient_output, axis=sum_axes)
        
        dx_norm = gradient_output * self.gamma
        
        dvar = np.sum(dx_norm * self.x_centered * -0.5 * (self.var + self.epsilon) ** (-1.5), axis=axes, keepdims=True)
        
        dmean = np.sum(dx_norm * -1 / self.std, axis=axes, keepdims=True)
        dmean += dvar * np.mean(-2 * self.x_centered, axis=axes, keepdims=True)
        
        dx = dx_norm / self.std
        dx += dvar * 2 * self.x_centered / n
        dx += dmean / n
        
        return dx
