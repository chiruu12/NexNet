import numpy as np


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention.
    
    The core attention mechanism: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    
    Supports optional masking for:
    - Causal/autoregressive attention (GPT-style)
    - Padding mask for variable length sequences
    """
    
    def __init__(self, dropout_rate=0.0):
        """
        Initialize Scaled Dot-Product Attention.
        
        Args:
            dropout_rate: Dropout rate applied to attention weights.
        """
        self.dropout_rate = dropout_rate
        self.training = True
        
    def forward(self, query, key, value, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor of shape (batch, seq_len_q, d_k).
            key: Key tensor of shape (batch, seq_len_k, d_k).
            value: Value tensor of shape (batch, seq_len_k, d_v).
            mask: Optional mask tensor. Use -inf for positions to mask.
                  Shape: (batch, 1, seq_len_q, seq_len_k) or broadcastable.
            
        Returns:
            output: Attention output of shape (batch, seq_len_q, d_v).
            attention_weights: Attention weights of shape (batch, seq_len_q, seq_len_k).
        """
        self.query = query
        self.key = key
        self.value = value
        self.mask = mask
        
        self.d_k = query.shape[-1]
        self.scale = np.sqrt(self.d_k)
        
        scores = np.matmul(query, key.transpose(0, 2, 1)) / self.scale
        
        if mask is not None:
            scores = scores + mask
            
        self.attention_weights = softmax(scores, axis=-1)
        
        if self.training and self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*self.attention_weights.shape) > self.dropout_rate).astype(np.float64)
            self.attention_weights_dropped = self.attention_weights * self.dropout_mask / (1 - self.dropout_rate)
        else:
            self.attention_weights_dropped = self.attention_weights
            
        self.output = np.matmul(self.attention_weights_dropped, value)
        
        return self.output, self.attention_weights
    
    def backward(self, gradient_output):
        """
        Backward pass of scaled dot-product attention.
        
        Args:
            gradient_output: Gradient from the next layer, shape (batch, seq_len_q, d_v).
            
        Returns:
            dquery: Gradient w.r.t. query.
            dkey: Gradient w.r.t. key.
            dvalue: Gradient w.r.t. value.
        """
        dvalue = np.matmul(self.attention_weights_dropped.transpose(0, 2, 1), gradient_output)
        
        d_attn_weights = np.matmul(gradient_output, self.value.transpose(0, 2, 1))
        
        if self.training and self.dropout_rate > 0:
            d_attn_weights = d_attn_weights * self.dropout_mask / (1 - self.dropout_rate)
            
        d_scores = self.attention_weights * (d_attn_weights - np.sum(d_attn_weights * self.attention_weights, axis=-1, keepdims=True))
        
        d_scores = d_scores / self.scale
        
        dquery = np.matmul(d_scores, self.key)
        dkey = np.matmul(d_scores.transpose(0, 2, 1), self.query)
        
        return dquery, dkey, dvalue
    
    def train(self):
        """Set to training mode."""
        self.training = True
        
    def eval(self):
        """Set to evaluation mode."""
        self.training = False


class MultiHeadAttention:
    """
    Multi-Head Attention.
    
    Runs multiple attention heads in parallel, allowing the model to
    jointly attend to information from different representation subspaces.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
    where head_i = Attention(Q * W_Q_i, K * W_K_i, V * W_V_i)
    """
    
    def __init__(self, d_model, num_heads, dropout_rate=0.0):
        """
        Initialize Multi-Head Attention.
        
        Args:
            d_model: Dimension of the model.
            num_heads: Number of attention heads.
            dropout_rate: Dropout rate for attention weights.
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_rate = dropout_rate
        self.training = True
        
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale
        
        self.b_Q = np.zeros((1, d_model))
        self.b_K = np.zeros((1, d_model))
        self.b_V = np.zeros((1, d_model))
        self.b_O = np.zeros((1, d_model))
        
        self._init_gradients()
        
        self.attention = ScaledDotProductAttention(dropout_rate)
        
    def _init_gradients(self):
        """Initialize gradient accumulators."""
        self.dW_Q = np.zeros_like(self.W_Q)
        self.dW_K = np.zeros_like(self.W_K)
        self.dW_V = np.zeros_like(self.W_V)
        self.dW_O = np.zeros_like(self.W_O)
        
        self.db_Q = np.zeros_like(self.b_Q)
        self.db_K = np.zeros_like(self.b_K)
        self.db_V = np.zeros_like(self.b_V)
        self.db_O = np.zeros_like(self.b_O)
        
    def _split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        
        Args:
            x: Tensor of shape (batch, seq_len, d_model).
            
        Returns:
            Tensor of shape (batch, num_heads, seq_len, d_k).
        """
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def _combine_heads(self, x):
        """
        Combine heads back to original shape.
        
        Args:
            x: Tensor of shape (batch, num_heads, seq_len, d_k).
            
        Returns:
            Tensor of shape (batch, seq_len, d_model).
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.d_model)
        
    def forward(self, query, key, value, mask=None):
        """
        Forward pass of Multi-Head Attention.
        
        Args:
            query: Query tensor of shape (batch, seq_len_q, d_model).
            key: Key tensor of shape (batch, seq_len_k, d_model).
            value: Value tensor of shape (batch, seq_len_k, d_model).
            mask: Optional attention mask.
            
        Returns:
            output: Attention output of shape (batch, seq_len_q, d_model).
            attention_weights: Attention weights from all heads.
        """
        self.query_input = query
        self.key_input = key
        self.value_input = value
        
        batch_size = query.shape[0]
        
        Q = np.dot(query.reshape(-1, self.d_model), self.W_Q).reshape(batch_size, -1, self.d_model) + self.b_Q
        K = np.dot(key.reshape(-1, self.d_model), self.W_K).reshape(batch_size, -1, self.d_model) + self.b_K
        V = np.dot(value.reshape(-1, self.d_model), self.W_V).reshape(batch_size, -1, self.d_model) + self.b_V
        
        self.Q_projected = Q
        self.K_projected = K
        self.V_projected = V
        
        Q_heads = self._split_heads(Q)
        K_heads = self._split_heads(K)
        V_heads = self._split_heads(V)
        
        self.attention_outputs = []
        self.attention_weights_list = []
        
        Q_heads_flat = Q_heads.reshape(batch_size * self.num_heads, -1, self.d_k)
        K_heads_flat = K_heads.reshape(batch_size * self.num_heads, -1, self.d_k)
        V_heads_flat = V_heads.reshape(batch_size * self.num_heads, -1, self.d_k)
        
        if mask is not None:
            mask_expanded = np.tile(mask, (1, self.num_heads, 1, 1)).reshape(batch_size * self.num_heads, mask.shape[2], mask.shape[3])
        else:
            mask_expanded = None
            
        attn_output, attn_weights = self.attention.forward(Q_heads_flat, K_heads_flat, V_heads_flat, mask_expanded)
        
        attn_output = attn_output.reshape(batch_size, self.num_heads, -1, self.d_k)
        self.attn_output_combined = self._combine_heads(attn_output)
        
        self.output = np.dot(self.attn_output_combined.reshape(-1, self.d_model), self.W_O).reshape(batch_size, -1, self.d_model) + self.b_O
        
        self.attention_weights = attn_weights.reshape(batch_size, self.num_heads, -1, attn_weights.shape[-1])
        
        return self.output, self.attention_weights
    
    def backward(self, gradient_output):
        """
        Backward pass of Multi-Head Attention.
        
        Args:
            gradient_output: Gradient from the next layer.
            
        Returns:
            dquery: Gradient w.r.t. query input.
            dkey: Gradient w.r.t. key input.
            dvalue: Gradient w.r.t. value input.
        """
        batch_size = gradient_output.shape[0]
        seq_len_q = gradient_output.shape[1]
        seq_len_k = self.key_input.shape[1]
        
        grad_flat = gradient_output.reshape(-1, self.d_model)
        self.dW_O = np.dot(self.attn_output_combined.reshape(-1, self.d_model).T, grad_flat)
        self.db_O = np.sum(grad_flat, axis=0, keepdims=True)
        
        d_attn_combined = np.dot(grad_flat, self.W_O.T).reshape(batch_size, seq_len_q, self.d_model)
        
        d_attn_heads = self._split_heads(d_attn_combined)
        d_attn_flat = d_attn_heads.reshape(batch_size * self.num_heads, seq_len_q, self.d_k)
        
        dQ_flat, dK_flat, dV_flat = self.attention.backward(d_attn_flat)
        
        dQ_heads = dQ_flat.reshape(batch_size, self.num_heads, seq_len_q, self.d_k)
        dK_heads = dK_flat.reshape(batch_size, self.num_heads, seq_len_k, self.d_k)
        dV_heads = dV_flat.reshape(batch_size, self.num_heads, seq_len_k, self.d_k)
        
        dQ = self._combine_heads(dQ_heads)
        dK = self._combine_heads(dK_heads)
        dV = self._combine_heads(dV_heads)
        
        dQ_flat = dQ.reshape(-1, self.d_model)
        dK_flat = dK.reshape(-1, self.d_model)
        dV_flat = dV.reshape(-1, self.d_model)
        
        self.dW_Q = np.dot(self.query_input.reshape(-1, self.d_model).T, dQ_flat)
        self.dW_K = np.dot(self.key_input.reshape(-1, self.d_model).T, dK_flat)
        self.dW_V = np.dot(self.value_input.reshape(-1, self.d_model).T, dV_flat)
        
        self.db_Q = np.sum(dQ_flat, axis=0, keepdims=True)
        self.db_K = np.sum(dK_flat, axis=0, keepdims=True)
        self.db_V = np.sum(dV_flat, axis=0, keepdims=True)
        
        dquery = np.dot(dQ_flat, self.W_Q.T).reshape(batch_size, seq_len_q, self.d_model)
        dkey = np.dot(dK_flat, self.W_K.T).reshape(batch_size, seq_len_k, self.d_model)
        dvalue = np.dot(dV_flat, self.W_V.T).reshape(batch_size, seq_len_k, self.d_model)
        
        return dquery, dkey, dvalue
    
    def train(self):
        """Set to training mode."""
        self.training = True
        self.attention.train()
        
    def eval(self):
        """Set to evaluation mode."""
        self.training = False
        self.attention.eval()


def create_causal_mask(seq_length):
    """
    Create a causal (autoregressive) mask for decoder self-attention.
    
    Prevents positions from attending to subsequent positions.
    
    Args:
        seq_length: Length of the sequence.
        
    Returns:
        Mask tensor of shape (1, 1, seq_length, seq_length) with -inf for masked positions.
    """
    mask = np.triu(np.ones((seq_length, seq_length)), k=1)
    mask = mask * -1e9
    return mask[np.newaxis, np.newaxis, :, :]


def create_padding_mask(seq, pad_idx=0):
    """
    Create a padding mask to ignore pad tokens.
    
    Args:
        seq: Input sequence of shape (batch, seq_length).
        pad_idx: Index of the padding token.
        
    Returns:
        Mask tensor of shape (batch, 1, 1, seq_length) with -inf for pad positions.
    """
    mask = (seq == pad_idx).astype(np.float64)
    mask = mask * -1e9
    return mask[:, np.newaxis, np.newaxis, :]
