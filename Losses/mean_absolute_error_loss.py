import numpy as np
class MAE:
    def forward(self, predictions,targets):
        """
        Perform the forward pass of the Mean Absolute Error (MAE) Loss function.

        Args:
            targets : True labels
            predictions : Predicted values

        Returns:
            The computed MAE loss.
        """
        self.predictions = predictions
        self.targets = targets
        # formulae is summation of mod of all the differences n then divided by number of differences (difference btw prediction and target value)
        # i.e it is = sum(abs( difference ) )/number of differences 
        self.loss = np.mean(np.abs(predictions - targets))
        return self.loss

    def backward(self):
        """
        Perform the backward pass of the Mean Absolute Error (MAE) Loss function.

        Returns:
            Gradient of the loss with respect to the predictions
        """
        # Gradient for MAE loss is either +1 or -1 depending on the sign of the error
        return 1/self.targets.size if self.predictions > self.targets else -1/self.targets.size
