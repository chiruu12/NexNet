import numpy as np


class SinusoidalPositionalEncoding:
    """
    Sinusoidal Positional Encoding.
    
    Adds positional information to embeddings using sine and cosine functions
    of different frequencies. This is the original positional encoding from
    "Attention Is All You Need" paper.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
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
        pe = np.zeros((self.max_seq_length, self.d_model))
        
        position = np.arange(self.max_seq_length)[:, np.newaxis]
        
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model).
            
        Returns:
            Input with positional encoding added.
        """
        self.input = x
        seq_length = x.shape[1]
        
        self.output = x + self.pe[:seq_length]
        
        if self.training and self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*self.output.shape) > self.dropout_rate).astype(np.float64)
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
        
        self.pe = np.random.randn(max_seq_length, d_model) * 0.02
        self.dpe = np.zeros_like(self.pe)
        
    def forward(self, x):
        """
        Add learned positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model).
            
        Returns:
            Input with positional encoding added.
        """
        self.input = x
        self.seq_length = x.shape[1]
        
        self.output = x + self.pe[:self.seq_length]
        
        if self.training and self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*self.output.shape) > self.dropout_rate).astype(np.float64)
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
            
        self.dpe = np.zeros_like(self.pe)
        self.dpe[:self.seq_length] = np.sum(gradient_output, axis=0)
        
        return gradient_output
    
    def train(self):
        """Set to training mode."""
        self.training = True
        
    def eval(self):
        """Set to evaluation mode."""
        self.training = False
