from core.backend import (
    get_array_module, zeros, zeros_like, random_randn, sin, cos, exp, log, 
    arange, sum as xp_sum, random_rand
)


class SinusoidalPositionalEncoding:
    """
    Sinusoidal Positional Encoding.
    
    Adds positional information to embeddings using sine and cosine functions
    of different frequencies. This is the original positional encoding from
    "Attention Is All You Need" paper.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Supports both NumPy and CuPy backends transparently.
    """
    
    def __init__(self, d_model, max_seq_length=5000, dropout_rate=0.1):
        """
        Initialize Sinusoidal Positional Encoding.
        
        Args:
            d_model: Dimension of the model (embedding size).
            max_seq_length: Maximum sequence length to precompute encodings for.
            dropout_rate: Dropout rate applied after adding positional encoding.
        """
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout_rate
        self.training = True
        
        self.pe = self._create_positional_encoding()
        
    def _create_positional_encoding(self):
        """Create the positional encoding matrix."""
        pe = zeros((self.max_seq_length, self.d_model))
        
        position = arange(self.max_seq_length)[:, None]
        
        div_term = exp(arange(0, self.d_model, 2) * -(log(10000.0) / self.d_model))
        
        pe[:, 0::2] = sin(position * div_term)
        pe[:, 1::2] = cos(position * div_term)
        
        return pe
    
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model).
            
        Returns:
            Input with positional encoding added.
        """
        xp = get_array_module(x)
        self.input = x
        seq_length = x.shape[1]
        
        self.output = x + self.pe[:seq_length]
        
        if self.training and self.dropout_rate > 0:
            self.dropout_mask = (random_rand(*self.output.shape) > self.dropout_rate).astype(xp.float64)
            self.output = self.output * self.dropout_mask / (1 - self.dropout_rate)
            
        return self.output
    
    def backward(self, gradient_output):
        """
        Backward pass (positional encoding has no learnable parameters).
        
        Args:
            gradient_output: Gradient from the next layer.
            
        Returns:
            Gradient with respect to input.
        """
        if self.training and self.dropout_rate > 0:
            return gradient_output * self.dropout_mask / (1 - self.dropout_rate)
        return gradient_output
    
    def train(self):
        """Set to training mode."""
        self.training = True
        
    def eval(self):
        """Set to evaluation mode."""
        self.training = False


class LearnedPositionalEncoding:
    """
    Learned Positional Encoding.
    
    Uses learnable position embeddings instead of fixed sinusoidal patterns.
    This is the approach used in GPT and BERT models.
    
    Supports both NumPy and CuPy backends transparently.
    """
    
    def __init__(self, d_model, max_seq_length=512, dropout_rate=0.1):
        """
        Initialize Learned Positional Encoding.
        
        Args:
            d_model: Dimension of the model (embedding size).
            max_seq_length: Maximum sequence length.
            dropout_rate: Dropout rate applied after adding positional encoding.
        """
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout_rate
        self.training = True
        
        self.pe = random_randn(max_seq_length, d_model) * 0.02
        self.dpe = zeros_like(self.pe)
        
    def forward(self, x):
        """
        Add learned positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model).
            
        Returns:
            Input with positional encoding added.
        """
        xp = get_array_module(x)
        self.input = x
        self.seq_length = x.shape[1]
        
        self.output = x + self.pe[:self.seq_length]
        
        if self.training and self.dropout_rate > 0:
            self.dropout_mask = (random_rand(*self.output.shape) > self.dropout_rate).astype(xp.float64)
            self.output = self.output * self.dropout_mask / (1 - self.dropout_rate)
            
        return self.output
    
    def backward(self, gradient_output):
        """
        Backward pass for learned positional encoding.
        
        Args:
            gradient_output: Gradient from the next layer.
            
        Returns:
            Gradient with respect to input.
        """
        if self.training and self.dropout_rate > 0:
            gradient_output = gradient_output * self.dropout_mask / (1 - self.dropout_rate)
            
        self.dpe = zeros_like(self.pe)
        self.dpe[:self.seq_length] = xp_sum(gradient_output, axis=0)
        
        return gradient_output
    
    def train(self):
        """Set to training mode."""
        self.training = True
        
    def eval(self):
        """Set to evaluation mode."""
        self.training = False
