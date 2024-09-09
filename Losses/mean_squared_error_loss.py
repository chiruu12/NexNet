import numpy as np                          
class MSE:
    def forward(self, predictions, targets):
        """
        Perform the forward pass of the Mean Squared error Loss function.

        Args:
            targets : True labels, one-hot encoded 
            predictions : Predicted probabilities 

        Returns:
            The computed cross-entropy loss.
        """
        self.predictions = predictions
        self.targets = targets
        # formulae is summation of square of all the differences n then divided by number of differences (difference btw prediction and target value)
        # i.e it is = sum(( difference ) ** 2 )/number of differences 
        self.loss = np.mean((predictions - targets) ** 2)
        return self.loss

    def backward(self):
        """
        Perform the backward pass of the Mean Squared error Loss function.

        Returns:
            Gradient of the loss with respect to the predictions
        """
        # Gradients for MSE
        grad_input = 2*(self.predictions - self.targets)/self.targets.size
        return grad_input