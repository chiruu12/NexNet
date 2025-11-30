import math
from core.backend import get_array_module, exp, sqrt, tanh, erf


class GELU:
    """
    Gaussian Error Linear Unit activation function.
    
    Used in GPT, BERT, and other transformer models. Provides smooth
    non-linearity that allows small negative values to pass through.
    
    GELU(x) = x * Φ(x) where Φ is the CDF of standard normal distribution.
    
    Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    
    def __init__(self, approximate=True):
        """
        Initialize the GELU activation.
        
        Args:
            approximate: If True, use tanh approximation (faster).
                        If False, use exact computation with erf.
        """
        self.approximate = approximate
        
    def forward(self, x):
        """
        Forward pass of GELU activation.
        
        Args:
            x: Input tensor of any shape.
            
        Returns:
            Output tensor with GELU applied element-wise.
        """
        self.input = x
        
        if self.approximate:
            self.output = 0.5 * x * (1 + tanh(sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
        else:
            self.output = 0.5 * x * (1 + erf(x / sqrt(2)))
            
        return self.output
    
    def backward(self, gradient_output):
        """
        Backward pass of GELU activation.
        
        Args:
            gradient_output: Gradient from the next layer.
            
        Returns:
            Gradient with respect to input.
        """
        x = self.input
        
        if self.approximate:
            tanh_arg = sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)
            tanh_val = tanh(tanh_arg)
            sech2 = 1 - tanh_val ** 2
            
            dtanh = sqrt(2 / math.pi) * (1 + 3 * 0.044715 * x ** 2)
            
            dgelu = 0.5 * (1 + tanh_val) + 0.5 * x * sech2 * dtanh
        else:
            phi = 0.5 * (1 + erf(x / sqrt(2)))
            pdf = exp(-0.5 * x ** 2) / sqrt(2 * math.pi)
            dgelu = phi + x * pdf
            
        self.diffv = gradient_output * dgelu
        return self.diffv
