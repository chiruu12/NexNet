from core.backend import get_array_module, random_randn, sqrt, zeros_like


class Embedding:
    """
    Embedding Layer.
    
    Converts integer indices into dense vectors of fixed size.
    Commonly used as the first layer in NLP models to convert
    word indices into word embeddings.
    
    Input shape: (batch_size, sequence_length) of integer indices
    Output shape: (batch_size, sequence_length, embedding_dim)
    """
    
    def __init__(self, vocab_size, embedding_dim, initialization='random'):
        """
        Initialize the Embedding layer.
        
        Args:
            vocab_size: Size of the vocabulary (maximum integer index + 1).
            embedding_dim: Dimension of the embedding vectors.
            initialization: Weight initialization method ('random', 'xavier').
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        if initialization == 'xavier':
            scale = sqrt(2.0 / (vocab_size + embedding_dim))
        else:
            scale = 0.01
            
        self.W = random_randn(vocab_size, embedding_dim) * scale
        self.dW = zeros_like(self.W)
        
    def forward(self, x):
        """
        Forward pass of the Embedding layer.
        
        Args:
            x: Input tensor of integer indices (batch_size, sequence_length).
            
        Returns:
            Embedded vectors (batch_size, sequence_length, embedding_dim).
        """
        self.input_indices = x
        self.output = self.W[x]
        return self.output
    
    def backward(self, gradient_output):
        """
        Backward pass of the Embedding layer.
        
        Args:
            gradient_output: Gradient from the next layer
                            (batch_size, sequence_length, embedding_dim).
            
        Returns:
            None (no gradient for integer inputs).
        """
        self.dW = zeros_like(self.W)
        
        batch_size, seq_length, _ = gradient_output.shape
        
        for b in range(batch_size):
            for t in range(seq_length):
                idx = self.input_indices[b, t]
                self.dW[idx] += gradient_output[b, t]
                
        return None
    
    def load_pretrained(self, embeddings, freeze=False):
        """
        Load pretrained embeddings.
        
        Args:
            embeddings: Pretrained embedding matrix (vocab_size, embedding_dim).
            freeze: If True, don't update embeddings during training.
        """
        assert embeddings.shape == self.W.shape, \
            f"Shape mismatch: expected {self.W.shape}, got {embeddings.shape}"
        self.W = embeddings.copy()
        self.freeze = freeze
        
    def get_embedding(self, index):
        """Get embedding vector for a specific index."""
        return self.W[index]
