import numpy as np  

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