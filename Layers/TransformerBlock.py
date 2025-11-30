from core.backend import (
    get_array_module, zeros, zeros_like, random_randn, sqrt, dot, 
    tanh, maximum, sum as xp_sum, random_rand
)
from .Attention import MultiHeadAttention, create_causal_mask
from .LayerNorm import LayerNorm


class FeedForward:
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = max(0, xW1 + b1)W2 + b2  (with ReLU)
    or
    FFN(x) = GELU(xW1 + b1)W2 + b2    (with GELU, used in GPT)
    
    Typically d_ff = 4 * d_model.
    
    Supports both NumPy and CuPy backends transparently.
    """
    
    def __init__(self, d_model, d_ff, dropout_rate=0.1, activation='gelu'):
        """
        Initialize Feed-Forward Network.
        
        Args:
            d_model: Input and output dimension.
            d_ff: Hidden layer dimension (typically 4 * d_model).
            dropout_rate: Dropout rate after activation.
            activation: Activation function ('gelu' or 'relu').
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.activation_type = activation
        self.training = True
        
        scale1 = sqrt(2.0 / (d_model + d_ff))
        scale2 = sqrt(2.0 / (d_ff + d_model))
        
        self.W1 = random_randn(d_model, d_ff) * scale1
        self.b1 = zeros((1, d_ff))
        self.W2 = random_randn(d_ff, d_model) * scale2
        self.b2 = zeros((1, d_model))
        
        self._init_gradients()
        
    def _init_gradients(self):
        """Initialize gradient accumulators."""
        self.dW1 = zeros_like(self.W1)
        self.db1 = zeros_like(self.b1)
        self.dW2 = zeros_like(self.W2)
        self.db2 = zeros_like(self.b2)
        
    def _gelu(self, x):
        """GELU activation."""
        import math
        return 0.5 * x * (1 + tanh(sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
    
    def _gelu_backward(self, x, grad):
        """GELU backward."""
        import math
        tanh_arg = sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)
        tanh_val = tanh(tanh_arg)
        sech2 = 1 - tanh_val ** 2
        dtanh = sqrt(2 / math.pi) * (1 + 3 * 0.044715 * x ** 2)
        dgelu = 0.5 * (1 + tanh_val) + 0.5 * x * sech2 * dtanh
        return grad * dgelu
        
    def forward(self, x):
        """
        Forward pass of Feed-Forward Network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        xp = get_array_module(x)
        self.input = x
        batch_size, seq_len, _ = x.shape
        
        x_flat = x.reshape(-1, self.d_model)
        
        self.hidden_pre = dot(x_flat, self.W1) + self.b1
        
        if self.activation_type == 'gelu':
            self.hidden = self._gelu(self.hidden_pre)
        else:
            self.hidden = maximum(0, self.hidden_pre)
            
        if self.training and self.dropout_rate > 0:
            self.dropout_mask = (random_rand(*self.hidden.shape) > self.dropout_rate).astype(xp.float64)
            self.hidden_dropped = self.hidden * self.dropout_mask / (1 - self.dropout_rate)
        else:
            self.hidden_dropped = self.hidden
            
        output_flat = dot(self.hidden_dropped, self.W2) + self.b2
        
        self.output = output_flat.reshape(batch_size, seq_len, self.d_model)
        
        return self.output
    
    def backward(self, gradient_output):
        """
        Backward pass of Feed-Forward Network.
        
        Args:
            gradient_output: Gradient from the next layer.
            
        Returns:
            Gradient with respect to input.
        """
        xp = get_array_module(gradient_output)
        batch_size, seq_len, _ = gradient_output.shape
        grad_flat = gradient_output.reshape(-1, self.d_model)
        
        self.dW2 = dot(self.hidden_dropped.T, grad_flat)
        self.db2 = xp_sum(grad_flat, axis=0, keepdims=True)
        
        d_hidden_dropped = dot(grad_flat, self.W2.T)
        
        if self.training and self.dropout_rate > 0:
            d_hidden = d_hidden_dropped * self.dropout_mask / (1 - self.dropout_rate)
        else:
            d_hidden = d_hidden_dropped
            
        if self.activation_type == 'gelu':
            d_hidden_pre = self._gelu_backward(self.hidden_pre, d_hidden)
        else:
            d_hidden_pre = d_hidden * (self.hidden_pre > 0).astype(xp.float64)
            
        self.dW1 = dot(self.input.reshape(-1, self.d_model).T, d_hidden_pre)
        self.db1 = xp_sum(d_hidden_pre, axis=0, keepdims=True)
        
        dx_flat = dot(d_hidden_pre, self.W1.T)
        dx = dx_flat.reshape(batch_size, seq_len, self.d_model)
        
        return dx
    
    def train(self):
        """Set to training mode."""
        self.training = True
        
    def eval(self):
        """Set to evaluation mode."""
        self.training = False


class TransformerDecoderBlock:
    """
    Transformer Decoder Block (GPT-style).
    
    Architecture:
    1. Masked Multi-Head Self-Attention + Residual + LayerNorm
    2. Feed-Forward Network + Residual + LayerNorm
    
    Uses Pre-LN architecture (LayerNorm before attention/FFN) for better training stability.
    
    Supports both NumPy and CuPy backends transparently.
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, pre_norm=True):
        """
        Initialize Transformer Decoder Block.
        
        Args:
            d_model: Dimension of the model.
            num_heads: Number of attention heads.
            d_ff: Dimension of feed-forward hidden layer.
            dropout_rate: Dropout rate.
            pre_norm: If True, use Pre-LN (LayerNorm before sublayers).
                     If False, use Post-LN (LayerNorm after sublayers).
        """
        self.d_model = d_model
        self.pre_norm = pre_norm
        self.dropout_rate = dropout_rate
        self.training = True
        
        self.ln1 = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout_rate)
        
    def forward(self, x, mask=None):
        """
        Forward pass of Transformer Decoder Block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional attention mask (for causal masking).
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        xp = get_array_module(x)
        self.input = x
        
        if self.pre_norm:
            x_norm1 = self.ln1.forward(x)
            attn_output, self.attn_weights = self.attention.forward(x_norm1, x_norm1, x_norm1, mask)
            
            if self.training and self.dropout_rate > 0:
                self.dropout_mask1 = (random_rand(*attn_output.shape) > self.dropout_rate).astype(xp.float64)
                attn_output = attn_output * self.dropout_mask1 / (1 - self.dropout_rate)
                
            self.residual1 = x + attn_output
            
            x_norm2 = self.ln2.forward(self.residual1)
            ffn_output = self.ffn.forward(x_norm2)
            
            if self.training and self.dropout_rate > 0:
                self.dropout_mask2 = (random_rand(*ffn_output.shape) > self.dropout_rate).astype(xp.float64)
                ffn_output = ffn_output * self.dropout_mask2 / (1 - self.dropout_rate)
                
            self.output = self.residual1 + ffn_output
            
        else:
            attn_output, self.attn_weights = self.attention.forward(x, x, x, mask)
            
            if self.training and self.dropout_rate > 0:
                self.dropout_mask1 = (random_rand(*attn_output.shape) > self.dropout_rate).astype(xp.float64)
                attn_output = attn_output * self.dropout_mask1 / (1 - self.dropout_rate)
                
            self.residual1 = self.ln1.forward(x + attn_output)
            
            ffn_output = self.ffn.forward(self.residual1)
            
            if self.training and self.dropout_rate > 0:
                self.dropout_mask2 = (random_rand(*ffn_output.shape) > self.dropout_rate).astype(xp.float64)
                ffn_output = ffn_output * self.dropout_mask2 / (1 - self.dropout_rate)
                
            self.output = self.ln2.forward(self.residual1 + ffn_output)
            
        return self.output
    
    def backward(self, gradient_output):
        """
        Backward pass of Transformer Decoder Block.
        
        Args:
            gradient_output: Gradient from the next layer.
            
        Returns:
            Gradient with respect to input.
        """
        if self.pre_norm:
            d_residual1 = gradient_output
            
            if self.training and self.dropout_rate > 0:
                d_ffn_out = gradient_output * self.dropout_mask2 / (1 - self.dropout_rate)
            else:
                d_ffn_out = gradient_output
                
            d_x_norm2 = self.ffn.backward(d_ffn_out)
            d_residual1 = d_residual1 + self.ln2.backward(d_x_norm2)
            
            d_input = d_residual1
            
            if self.training and self.dropout_rate > 0:
                d_attn_out = d_residual1 * self.dropout_mask1 / (1 - self.dropout_rate)
            else:
                d_attn_out = d_residual1
                
            dq, dk, dv = self.attention.backward(d_attn_out)
            d_x_norm1 = dq + dk + dv
            d_input = d_input + self.ln1.backward(d_x_norm1)
            
        else:
            d_ln2 = self.ln2.backward(gradient_output)
            
            d_residual1 = d_ln2
            
            if self.training and self.dropout_rate > 0:
                d_ffn_out = d_ln2 * self.dropout_mask2 / (1 - self.dropout_rate)
            else:
                d_ffn_out = d_ln2
                
            d_ln1_input = self.ffn.backward(d_ffn_out)
            
            d_ln1 = self.ln1.backward(d_residual1 + d_ln1_input)
            
            d_input = d_ln1
            
            if self.training and self.dropout_rate > 0:
                d_attn_out = d_ln1 * self.dropout_mask1 / (1 - self.dropout_rate)
            else:
                d_attn_out = d_ln1
                
            dq, dk, dv = self.attention.backward(d_attn_out)
            d_input = d_input + dq + dk + dv
            
        return d_input
    
    def train(self):
        """Set to training mode."""
        self.training = True
        self.attention.train()
        self.ffn.train()
        
    def eval(self):
        """Set to evaluation mode."""
        self.training = False
        self.attention.eval()
        self.ffn.eval()


class TransformerEncoderBlock:
    """
    Transformer Encoder Block (BERT-style).
    
    Similar to decoder block but without causal masking.
    Can attend to all positions in the sequence.
    
    Supports both NumPy and CuPy backends transparently.
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, pre_norm=True):
        """
        Initialize Transformer Encoder Block.
        
        Args:
            d_model: Dimension of the model.
            num_heads: Number of attention heads.
            d_ff: Dimension of feed-forward hidden layer.
            dropout_rate: Dropout rate.
            pre_norm: If True, use Pre-LN architecture.
        """
        self.decoder_block = TransformerDecoderBlock(d_model, num_heads, d_ff, dropout_rate, pre_norm)
        
    def forward(self, x, padding_mask=None):
        """
        Forward pass (no causal mask, just optional padding mask).
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            padding_mask: Optional padding mask.
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        return self.decoder_block.forward(x, mask=padding_mask)
    
    def backward(self, gradient_output):
        """Backward pass."""
        return self.decoder_block.backward(gradient_output)
    
    def train(self):
        """Set to training mode."""
        self.decoder_block.train()
        
    def eval(self):
        """Set to evaluation mode."""
        self.decoder_block.eval()
