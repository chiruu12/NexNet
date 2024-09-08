import numpy as np
#also known as cost functions when taken as sum of a batch as in sum of 1 to size of the batch losses
#can also be called as error function 

#all classification losses 
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

class BinaryCrossEntropyLoss:
    def __init__(self,epsilon=1e-5):
        """
        Initalizing the class 
        
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

    def backward(self):
        """
        Perform the backward pass of the MAE Loss function.

        Returns:
            Gradient of the loss with respect to the predictions
        """
        #calculating gradients loss 
        return 1/self.targets.size if self.predictions > self.targets else -1/self.targets.size

class CosineSimilarityLoss:
    def __init__(self, epsilon=1e-8):
        """
        Initialize the Cosine Similarity Loss class.

        Args:
            epsilon : to prevent division by zero and edge cases.
        """
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        """
        Compute the forward pass of the Cosine Similarity Loss function.

        Args:
            targets : True values, of shape (batch_size, num_features).
            predictions : Predicted values, of shape (batch_size, num_features).

        Returns:
            The computed Cosine Similarity Loss.
        """
        self.pred = predictions
        self.tar = targets
        # Compute the dot product
        self.dot = np.sum(self.pred *self.tar, axis=1)
        # Compute norms
        self.norm_pred = np.linalg.norm(self.predictions, axis=1)
        self.norm_tar = np.linalg.norm(self.targets, axis=1)

        # Compute cosine similarity and then the loss which is = 1 - similairty
        self.simi= self.dot/(self.norm_pred * self.norm_tar + self.epsilon)
        self.loss = 1 - np.mean(self.simi)
        return self.loss

    def backward(self):
        """
        Compute the backward pass of the Cosine Similarity Loss function.

        Returns:
            Gradient of the loss with respect to the predictions, and targets.
        """
        batch_size = self.predictions.shape[0]
        # calculate gradient
        grad_pred = (self.targets / (self.norm_tar + self.epsilon) - 
                            (self.dot/ (self.norm_tar**2 + self.epsilon))*(self.predictions /(self.norm_pred + self.epsilon)))/batch_size

        return grad_pred

    
    
# all the regression losses 
class PoissonLoss:
    def forward(self,k,_lambda_):
        """
        Perform the forward pass of the Huber loss function.

        Args:
            k : no. of events 
            _lambda_ : Average rate of events 
            # we have to write lambda as _lambda_ because of the lambda function!!

        Returns:
            The computed Poisson loss.
        """
        self._lambda_=_lambda_
        self.k=k
        # formula for probability is  = (lambda^k * e^(-lambda)) / k! 
        self.prob= ( _lambda_**self.k )*np.exp(-_lambda_)/np.maths.factorial(self.k)
        # loss is nothing but -log of the probability 
        self.loss=-np.log(self.prob)
        return self.loss
    def backward(self):
        """
        Compute the gradient of the Poisson loss with respect to lambda.

        Returns:
            Gradient of the Poisson loss with respect to lambda.
        """
        grad= -(self.k/self._lambda - 1)
        return grad
        
    
    
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

class MeanSquaredErrorLoss:
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
    
    
class MeanAbsoluteErrorLoss:
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
