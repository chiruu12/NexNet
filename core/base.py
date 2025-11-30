from abc import ABC, abstractmethod


class Layer(ABC):
    """
    Abstract base class for all neural network layers.
    
    All layers must implement forward and backward methods for
    the forward pass and backpropagation respectively.
    """
    
    @abstractmethod
    def forward(self, inputs):
        """
        Perform the forward pass of the layer.
        
        Args:
            inputs: Input data to the layer.
        
        Returns:
            Output of the layer.
        """
        pass
    
    @abstractmethod
    def backward(self, gradient_output):
        """
        Perform the backward pass of the layer.
        
        Args:
            gradient_output: Gradient of the loss with respect to the layer output.
        
        Returns:
            Gradient of the loss with respect to the layer input.
        """
        pass


class Activation(ABC):
    """
    Abstract base class for all activation functions.
    
    Activation functions transform the output of a layer
    to introduce non-linearity into the network.
    """
    
    @abstractmethod
    def forward(self, inputs):
        """
        Apply the activation function.
        
        Args:
            inputs: Input values to transform.
        
        Returns:
            Transformed values.
        """
        pass
    
    @abstractmethod
    def backward(self, gradient_output):
        """
        Compute the gradient of the activation function.
        
        Args:
            gradient_output: Gradient of the loss with respect to the activation output.
        
        Returns:
            Gradient of the loss with respect to the activation input.
        """
        pass


class Loss(ABC):
    """
    Abstract base class for all loss functions.
    
    Loss functions measure the difference between predicted
    and actual values during training.
    """
    
    @abstractmethod
    def forward(self, targets, predictions):
        """
        Compute the loss value.
        
        Args:
            targets: Ground truth values.
            predictions: Predicted values from the model.
        
        Returns:
            The computed loss value (scalar).
        """
        pass
    
    @abstractmethod
    def backward(self):
        """
        Compute the gradient of the loss with respect to predictions.
        
        Returns:
            Gradient of the loss with respect to the predictions.
        """
        pass


class Optimizer(ABC):
    """
    Abstract base class for all optimizers.
    
    Optimizers update the model parameters based on computed
    gradients to minimize the loss function.
    """
    
    @abstractmethod
    def step(self, layers):
        """
        Perform a single optimization step.
        
        Args:
            layers: List of layers containing parameters to update.
        """
        pass
    
    def zero_grad(self, layers):
        """
        Reset gradients of all layers to None.
        
        Args:
            layers: List of layers whose gradients should be reset.
        """
        for layer in layers:
            if hasattr(layer, 'dW'):
                layer.dW = None
            if hasattr(layer, 'db'):
                layer.db = None
