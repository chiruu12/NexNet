import numpy as np
class HuberLoss:
    def __init__(self,delta=1.0):
        """
        Initialize the class.

        Args:
            delta : The threshold where the loss transitions from quadratic to linear.
        """
        self.delta = delta

    def forward(self, predictions,targets):
        """
        Perform the forward pass of the Huber loss function.

        Args:
            predictions : Predicted values
            targets : True values 

        Returns:
            The computed Huber loss.
        """
        self.predictions = predictions
        self.targets = targets
        self.error = self.predictions - self.targets

        # calculating quadratic and linear part one of them will be returned as output. which one we are going to 
        # return depends on the value of delta we are using mse and mae in some sense here 
        quadratic_part = 0.5*(self.error ** 2)
        linear_part = self.delta*(np.abs(self.error) - 0.5*self.delta)
        return np.mean(quadratic_part) if np.abs(self.error) <= self.delta else linear_part

    def backward(self):
        """
        Perform the backward pass of the Huber loss function.

        Returns:
            Gradient of the loss with respect to the predictions
        """
        # Gradient for Huber loss
        return   self.error/self.targets.size if np.abs(self.error) <= self.delta else self.delta * np.sign(self.error)/self.targets.size
