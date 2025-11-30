import numpy as np


class HuberLoss:
    """
    Huber Loss (Smooth L1 Loss) for regression tasks.
    
    Combines the advantages of MSE and MAE: quadratic for small errors,
    linear for large errors, making it robust to outliers.
    """
    
    def __init__(self, delta=1.0):
        """
        Initialize the Huber Loss.
        
        Args:
            delta: Threshold where loss transitions from quadratic to linear.
        """
        self.delta = delta
        self.predictions = None
        self.targets = None
        self.error = None

    def forward(self, targets, predictions):
        """
        Compute the forward pass of the Huber Loss.
        
        Args:
            targets: True values of shape (batch_size,) or (batch_size, features).
            predictions: Predicted values of same shape as targets.
        
        Returns:
            The computed Huber loss (scalar).
        """
        self.predictions = predictions
        self.targets = targets
        self.error = self.predictions - self.targets
        
        abs_error = np.abs(self.error)
        quadratic = 0.5 * self.error ** 2
        linear = self.delta * (abs_error - 0.5 * self.delta)
        
        loss = np.where(abs_error <= self.delta, quadratic, linear)
        return np.mean(loss)

    def backward(self):
        """
        Compute the backward pass of the Huber Loss.
        
        Returns:
            Gradient of the loss with respect to the predictions.
        """
        abs_error = np.abs(self.error)
        batch_size = self.targets.size
        
        grad = np.where(
            abs_error <= self.delta,
            self.error / batch_size,
            self.delta * np.sign(self.error) / batch_size
        )
        return grad
