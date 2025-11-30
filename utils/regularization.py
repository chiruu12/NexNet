import numpy as np


class L1Regularization:
    """
    L1 (Lasso) Regularization.
    
    Adds a penalty equal to the absolute value of the magnitude of weights.
    This encourages sparse weights (many weights close to zero).
    
    The regularization term is: lambda * sum(|w|)
    
    Parameters
    ----------
    lambda_reg : float, optional
        Regularization strength. Default is 0.01.
        
    Examples
    --------
    >>> reg = L1Regularization(lambda_reg=0.001)
    >>> loss += reg.loss(model.layers)
    >>> reg.apply_gradients(model.layers)
    """
    
    def __init__(self, lambda_reg=0.01):
        """
        Initialize L1 regularization.
        
        Parameters
        ----------
        lambda_reg : float, optional
            Regularization strength. Default is 0.01.
        """
        self.lambda_reg = lambda_reg
    
    def loss(self, layers):
        """
        Compute the L1 regularization loss.
        
        Parameters
        ----------
        layers : list
            List of layer objects with weights.
            
        Returns
        -------
        float
            L1 regularization loss.
        """
        reg_loss = 0.0
        for layer in layers:
            if hasattr(layer, 'W') and layer.W is not None:
                reg_loss += np.sum(np.abs(layer.W))
        return self.lambda_reg * reg_loss
    
    def apply_gradients(self, layers):
        """
        Add L1 regularization gradients to layer gradients.
        
        Parameters
        ----------
        layers : list
            List of layer objects with weights and gradients.
        """
        for layer in layers:
            if hasattr(layer, 'W') and layer.W is not None and hasattr(layer, 'dW'):
                layer.dW += self.lambda_reg * np.sign(layer.W)


class L2Regularization:
    """
    L2 (Ridge) Regularization.
    
    Adds a penalty equal to the square of the magnitude of weights.
    This encourages small weights and helps prevent overfitting.
    
    The regularization term is: lambda * sum(w^2) / 2
    
    Parameters
    ----------
    lambda_reg : float, optional
        Regularization strength. Default is 0.01.
        
    Examples
    --------
    >>> reg = L2Regularization(lambda_reg=0.001)
    >>> loss += reg.loss(model.layers)
    >>> reg.apply_gradients(model.layers)
    """
    
    def __init__(self, lambda_reg=0.01):
        """
        Initialize L2 regularization.
        
        Parameters
        ----------
        lambda_reg : float, optional
            Regularization strength. Default is 0.01.
        """
        self.lambda_reg = lambda_reg
    
    def loss(self, layers):
        """
        Compute the L2 regularization loss.
        
        Parameters
        ----------
        layers : list
            List of layer objects with weights.
            
        Returns
        -------
        float
            L2 regularization loss.
        """
        reg_loss = 0.0
        for layer in layers:
            if hasattr(layer, 'W') and layer.W is not None:
                reg_loss += np.sum(layer.W ** 2)
        return 0.5 * self.lambda_reg * reg_loss
    
    def apply_gradients(self, layers):
        """
        Add L2 regularization gradients to layer gradients.
        
        Parameters
        ----------
        layers : list
            List of layer objects with weights and gradients.
        """
        for layer in layers:
            if hasattr(layer, 'W') and layer.W is not None and hasattr(layer, 'dW'):
                layer.dW += self.lambda_reg * layer.W


class ElasticNetRegularization:
    """
    Elastic Net Regularization.
    
    Combines L1 and L2 regularization with a mixing parameter.
    
    The regularization term is: alpha * l1_ratio * |w| + alpha * (1 - l1_ratio) * w^2 / 2
    
    Parameters
    ----------
    alpha : float, optional
        Overall regularization strength. Default is 0.01.
    l1_ratio : float, optional
        Mixing parameter between L1 and L2 (0 <= l1_ratio <= 1).
        l1_ratio = 1 gives L1, l1_ratio = 0 gives L2. Default is 0.5.
        
    Examples
    --------
    >>> reg = ElasticNetRegularization(alpha=0.001, l1_ratio=0.5)
    >>> loss += reg.loss(model.layers)
    >>> reg.apply_gradients(model.layers)
    """
    
    def __init__(self, alpha=0.01, l1_ratio=0.5):
        """
        Initialize Elastic Net regularization.
        
        Parameters
        ----------
        alpha : float, optional
            Regularization strength. Default is 0.01.
        l1_ratio : float, optional
            L1/L2 mixing ratio. Default is 0.5.
        """
        if not 0 <= l1_ratio <= 1:
            raise ValueError(f"l1_ratio must be between 0 and 1, got {l1_ratio}")
        self.alpha = alpha
        self.l1_ratio = l1_ratio
    
    def loss(self, layers):
        """
        Compute the Elastic Net regularization loss.
        
        Parameters
        ----------
        layers : list
            List of layer objects with weights.
            
        Returns
        -------
        float
            Elastic Net regularization loss.
        """
        l1_loss = 0.0
        l2_loss = 0.0
        for layer in layers:
            if hasattr(layer, 'W') and layer.W is not None:
                l1_loss += np.sum(np.abs(layer.W))
                l2_loss += np.sum(layer.W ** 2)
        
        return self.alpha * (self.l1_ratio * l1_loss + 0.5 * (1 - self.l1_ratio) * l2_loss)
    
    def apply_gradients(self, layers):
        """
        Add Elastic Net regularization gradients to layer gradients.
        
        Parameters
        ----------
        layers : list
            List of layer objects with weights and gradients.
        """
        for layer in layers:
            if hasattr(layer, 'W') and layer.W is not None and hasattr(layer, 'dW'):
                l1_grad = self.alpha * self.l1_ratio * np.sign(layer.W)
                l2_grad = self.alpha * (1 - self.l1_ratio) * layer.W
                layer.dW += l1_grad + l2_grad


class WeightDecay:
    """
    Weight Decay Regularization.
    
    Simple weight decay that directly decays the weights during optimization.
    This is equivalent to L2 regularization but applied differently.
    
    Parameters
    ----------
    decay : float, optional
        Weight decay factor. Default is 0.0001.
        
    Examples
    --------
    >>> decay = WeightDecay(decay=0.0001)
    >>> decay.apply(model.layers, learning_rate=0.01)
    """
    
    def __init__(self, decay=0.0001):
        """
        Initialize weight decay.
        
        Parameters
        ----------
        decay : float, optional
            Decay factor. Default is 0.0001.
        """
        self.decay = decay
    
    def apply(self, layers, learning_rate=1.0):
        """
        Apply weight decay to layer weights.
        
        Parameters
        ----------
        layers : list
            List of layer objects with weights.
        learning_rate : float, optional
            Learning rate for decay. Default is 1.0.
        """
        for layer in layers:
            if hasattr(layer, 'W') and layer.W is not None:
                layer.W -= learning_rate * self.decay * layer.W


class MaxNormConstraint:
    """
    Max-Norm Weight Constraint.
    
    Clips weights so their L2 norm doesn't exceed a maximum value.
    This can help prevent exploding weights and improve generalization.
    
    Parameters
    ----------
    max_norm : float, optional
        Maximum norm value. Default is 3.0.
    axis : int, optional
        Axis along which to compute the norm. Default is 0.
        
    Examples
    --------
    >>> constraint = MaxNormConstraint(max_norm=3.0)
    >>> constraint.apply(model.layers)
    """
    
    def __init__(self, max_norm=3.0, axis=0):
        """
        Initialize max-norm constraint.
        
        Parameters
        ----------
        max_norm : float, optional
            Maximum norm. Default is 3.0.
        axis : int, optional
            Axis for norm computation. Default is 0.
        """
        self.max_norm = max_norm
        self.axis = axis
    
    def apply(self, layers):
        """
        Apply max-norm constraint to layer weights.
        
        Parameters
        ----------
        layers : list
            List of layer objects with weights.
        """
        for layer in layers:
            if hasattr(layer, 'W') and layer.W is not None:
                norms = np.linalg.norm(layer.W, axis=self.axis, keepdims=True)
                desired_norms = np.clip(norms, 0, self.max_norm)
                scale = desired_norms / (norms + 1e-8)
                layer.W = layer.W * scale


class UnitNormConstraint:
    """
    Unit Norm Weight Constraint.
    
    Normalizes weights to have unit norm along a specified axis.
    
    Parameters
    ----------
    axis : int, optional
        Axis along which to normalize. Default is 0.
        
    Examples
    --------
    >>> constraint = UnitNormConstraint()
    >>> constraint.apply(model.layers)
    """
    
    def __init__(self, axis=0):
        """
        Initialize unit norm constraint.
        
        Parameters
        ----------
        axis : int, optional
            Axis for normalization. Default is 0.
        """
        self.axis = axis
    
    def apply(self, layers):
        """
        Apply unit norm constraint to layer weights.
        
        Parameters
        ----------
        layers : list
            List of layer objects with weights.
        """
        for layer in layers:
            if hasattr(layer, 'W') and layer.W is not None:
                norms = np.linalg.norm(layer.W, axis=self.axis, keepdims=True)
                layer.W = layer.W / (norms + 1e-8)
