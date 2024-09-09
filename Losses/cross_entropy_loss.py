import numpy as np
class CrossEntropyLoss:
    def __init__(self,epsilon=1e-5):
        """
        Initializing the class 
        
        Args:
            epsilon : used to avoid edge cases
        """
        self.epsilon =epsilon
    def forward(self, targets, predictions):
        """
        Perform the forward pass of the Cross-Entropy Loss function.

        Args:
            targets : True labels, one-hot encoded
            predictions : Predicted probabilities

        Returns:
            The computed cross-entropy loss.
        """
        p_max =np.max(predictions, axis=1,keepdims=True)
        exps = np.exp(predictions - p_max)
        self.softmax = exps/np.sum(exps, axis=1,keepdims=True)
        self.targets = targets

        # Computing the loss using the cross-entropy formula
        batch_size = predictions.shape[0]
        #here we are adding epsilon to avoid edge cases 
        self.loss =-np.sum(targets * np.log(self.softmax + self.epsilon))/batch_size 
        return self.loss

    def backward(self):
        """
        Perform the backward pass of the Cross-Entropy Loss function to compute gradients.

        Returns:
            Gradient of the loss with respect to the predictions
        """
        batch_size = self.targets.shape[0]
        # Computing gradient of the loss with respect to the predictions
        diff_predictions = (self.softmax -self.targets)/batch_size
        return diff_predictions