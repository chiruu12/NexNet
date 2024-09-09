import numpy as np 
class BinaryCrossEntropyLoss:
    def __init__(self,epsilon=1e-5):
        """
        Initializing the class 
        
        Args:
            epsilon : used to avoid edge cases
        """
        self.epsilon =epsilon
    def forward(self, targets, predictions):
        """
        Perform the forward pass of the Binary Cross-Entropy Loss function.

        Args:
            targets : True labels
            predictions : Predicted probabilities

        Returns:
            The computed binary cross-entropy loss.
        """
        # ensure predictions are in the range [epsilon, 1 - epsilon]
        predictions = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        
        # Computing the loss using the binary cross-entropy formula
        batch_size = targets.shape[0]
        self.loss = -np.sum(targets*np.log(predictions) + (1 -targets)*np.log(1 -predictions))/batch_size
        return self.loss

    def backward(self):
        """
        Perform the backward pass of the Binary Cross-Entropy Loss function to compute gradients.

        Returns:
            Gradient of the loss with respect to the predictions
        """
        batch_size = self.targets.shape[0]
        # Computing gradient of the loss with respect to the predictions
        diff_predictions = (self.predictions - self.targets)/(self.predictions*(1 - self.predictions)*batch_size)
        return diff_predictions