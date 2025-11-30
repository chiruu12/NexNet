import numpy as np


class Transformer:
    """
    Transformer model for sequence-to-sequence and language modeling tasks.
    
    This model provides a high-level interface for building GPT-style
    decoder-only transformers or encoder-decoder architectures.
    
    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    d_model : int
        Dimension of the model (embedding dimension).
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer blocks.
    d_ff : int, optional
        Dimension of feedforward network. Default is 4 * d_model.
    max_seq_len : int, optional
        Maximum sequence length. Default is 512.
    dropout : float, optional
        Dropout rate. Default is 0.1.
    causal : bool, optional
        Whether to use causal (autoregressive) attention. Default is True.
        
    Attributes
    ----------
    layers : list
        List of layers in the model.
    training : bool
        Whether the model is in training mode.
        
    Examples
    --------
    >>> from Models.Transformer import Transformer
    >>> 
    >>> model = Transformer(
    ...     vocab_size=10000,
    ...     d_model=256,
    ...     n_heads=8,
    ...     n_layers=6,
    ...     max_seq_len=512
    ... )
    >>> 
    >>> # Training
    >>> model.compile(optimizer, loss)
    >>> model.fit(X_train, y_train, epochs=10)
    >>> 
    >>> # Generation
    >>> generated = model.generate(start_tokens, max_length=100)
    """
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff=None,
                 max_seq_len=512, dropout=0.1, causal=True):
        """
        Initialize the Transformer model.
        
        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary.
        d_model : int
            Model dimension.
        n_heads : int
            Number of attention heads.
        n_layers : int
            Number of transformer blocks.
        d_ff : int, optional
            Feedforward dimension. Default is 4 * d_model.
        max_seq_len : int, optional
            Maximum sequence length. Default is 512.
        dropout : float, optional
            Dropout rate. Default is 0.1.
        causal : bool, optional
            Use causal attention. Default is True.
        """
        from Layers.Embedding import Embedding
        from Layers.PositionalEncoding import SinusoidalPositionalEncoding
        from Layers.TransformerBlock import TransformerDecoderBlock
        from Layers.Linear import Linear
        from Layers.LayerNorm import LayerNorm
        from Layers.Dropout import Dropout
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout
        self.causal = causal
        
        self.training = True
        self.optimizer = None
        self.loss_fn = None
        self._compiled = False
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        self.layers = []
        
        self.embedding = Embedding(vocab_size, d_model)
        self.layers.append(self.embedding)
        
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        self.layers.append(self.pos_encoding)
        
        self.transformer_blocks = []
        for _ in range(n_layers):
            block = TransformerDecoderBlock(d_model, n_heads, self.d_ff, dropout)
            self.transformer_blocks.append(block)
            self.layers.append(block)
        
        self.final_norm = LayerNorm(d_model)
        self.layers.append(self.final_norm)
        
        self.output_proj = Linear(d_model, vocab_size)
        self.layers.append(self.output_proj)
    
    def forward(self, x, mask=None):
        """
        Forward pass through the transformer.
        
        Parameters
        ----------
        x : ndarray
            Input token IDs of shape (batch_size, seq_len).
        mask : ndarray, optional
            Attention mask. Default is None.
            
        Returns
        -------
        ndarray
            Output logits of shape (batch_size, seq_len, vocab_size).
        """
        x = self.embedding.forward(x)
        
        x = x * np.sqrt(self.d_model)
        
        self.pos_encoding.training = self.training
        x = self.pos_encoding.forward(x)
        
        for block in self.transformer_blocks:
            block.training = self.training
            x = block.forward(x, mask)
        
        x = self.final_norm.forward(x)
        
        x = self.output_proj.forward(x)
        
        return x
    
    def backward(self, grad):
        """
        Backward pass through the transformer.
        
        Parameters
        ----------
        grad : ndarray
            Gradient from loss function.
            
        Returns
        -------
        ndarray
            Gradient with respect to input.
        """
        grad = self.output_proj.backward(grad)
        
        grad = self.final_norm.backward(grad)
        
        for block in reversed(self.transformer_blocks):
            grad = block.backward(grad)
        
        grad = self.pos_encoding.backward(grad)
        
        grad = grad * np.sqrt(self.d_model)
        
        grad = self.embedding.backward(grad)
        
        return grad
    
    def __call__(self, x, mask=None):
        """Make model callable."""
        return self.forward(x, mask)
    
    def train(self):
        """Set model to training mode."""
        self.training = True
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True
    
    def eval(self):
        """Set model to evaluation mode."""
        self.training = False
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False
    
    def compile(self, optimizer, loss):
        """
        Configure the model for training.
        
        Parameters
        ----------
        optimizer : Optimizer
            Optimizer for updating weights.
        loss : Loss
            Loss function.
        """
        self.optimizer = optimizer
        self.loss_fn = loss
        self._compiled = True
    
    def fit(self, X, y, epochs=10, batch_size=32, validation_data=None, verbose=True,
            callbacks=None, shuffle=True, clip_grad_norm=1.0, clip_grad_value=None):
        """
        Train the model.
        
        Parameters
        ----------
        X : ndarray
            Input token IDs of shape (n_samples, seq_len).
        y : ndarray
            Target token IDs of shape (n_samples, seq_len) or (n_samples,).
        epochs : int, optional
            Number of training epochs. Default is 10.
        batch_size : int, optional
            Mini-batch size. Default is 32.
        validation_data : tuple, optional
            Tuple of (X_val, y_val) for validation.
        verbose : bool, optional
            Print training progress. Default is True.
        callbacks : list, optional
            List of callback objects.
        shuffle : bool, optional
            Shuffle data each epoch. Default is True.
        clip_grad_norm : float, optional
            Maximum gradient norm. Default is 1.0.
        clip_grad_value : float, optional
            Maximum gradient value.
            
        Returns
        -------
        dict
            Training history.
        """
        if not self._compiled:
            raise RuntimeError("Model must be compiled before training. Call model.compile(optimizer, loss)")
        
        from utils.grad_clip import clip_grad_norm as cgn, clip_grad_value as cgv
        
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            self.train()
            
            if shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y
            
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                output = self.forward(X_batch)
                
                if y_batch.ndim == 2:
                    batch_size_actual, seq_len = y_batch.shape
                    output_flat = output.reshape(-1, self.vocab_size)
                    y_flat = y_batch.reshape(-1)
                    
                    loss = self.loss_fn.forward(y_flat, output_flat)
                    grad = self.loss_fn.backward()
                    grad = grad.reshape(batch_size_actual, seq_len, self.vocab_size)
                    
                    predictions = np.argmax(output, axis=-1)
                    epoch_correct += np.sum(predictions == y_batch)
                    epoch_total += y_batch.size
                else:
                    output_last = output[:, -1, :]
                    loss = self.loss_fn.forward(y_batch, output_last)
                    
                    grad_last = self.loss_fn.backward()
                    grad = np.zeros_like(output)
                    grad[:, -1, :] = grad_last
                    
                    predictions = np.argmax(output_last, axis=-1)
                    epoch_correct += np.sum(predictions == y_batch)
                    epoch_total += len(y_batch)
                
                self.backward(grad)
                
                if clip_grad_norm is not None:
                    cgn(self.layers, clip_grad_norm)
                if clip_grad_value is not None:
                    cgv(self.layers, clip_grad_value)
                
                self.optimizer.step(self.layers)
                
                epoch_loss += loss * (end_idx - start_idx)
            
            train_loss = epoch_loss / n_samples
            train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            val_str = ""
            if validation_data is not None:
                val_loss, val_acc = self.evaluate(validation_data[0], validation_data[1])
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                val_str = f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f}{val_str}")
            
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, {'loss': train_loss, 'acc': train_acc})
        
        return self.history
    
    def evaluate(self, X, y, batch_size=32):
        """
        Evaluate the model.
        
        Parameters
        ----------
        X : ndarray
            Input token IDs.
        y : ndarray
            Target token IDs.
        batch_size : int, optional
            Batch size for evaluation.
            
        Returns
        -------
        tuple
            (loss, accuracy)
        """
        self.eval()
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            
            output = self.forward(X_batch)
            
            if y_batch.ndim == 2:
                output_flat = output.reshape(-1, self.vocab_size)
                y_flat = y_batch.reshape(-1)
                loss = self.loss_fn.forward(y_flat, output_flat)
                
                predictions = np.argmax(output, axis=-1)
                total_correct += np.sum(predictions == y_batch)
                total_tokens += y_batch.size
            else:
                output_last = output[:, -1, :]
                loss = self.loss_fn.forward(y_batch, output_last)
                
                predictions = np.argmax(output_last, axis=-1)
                total_correct += np.sum(predictions == y_batch)
                total_tokens += len(y_batch)
            
            total_loss += loss * (end_idx - start_idx)
        
        return total_loss / n_samples, total_correct / total_tokens
    
    def generate(self, start_tokens, max_length, temperature=1.0, top_k=None, top_p=None):
        """
        Generate sequences autoregressively.
        
        Parameters
        ----------
        start_tokens : ndarray
            Starting token IDs of shape (batch_size, seq_len).
        max_length : int
            Maximum length of generated sequence.
        temperature : float, optional
            Sampling temperature. Default is 1.0.
        top_k : int, optional
            Sample from top-k tokens.
        top_p : float, optional
            Sample from tokens with cumulative probability <= top_p (nucleus sampling).
            
        Returns
        -------
        ndarray
            Generated token IDs of shape (batch_size, max_length).
        """
        self.eval()
        
        generated = start_tokens.copy()
        batch_size = start_tokens.shape[0]
        
        for _ in range(max_length - start_tokens.shape[1]):
            if generated.shape[1] > self.max_seq_len:
                input_tokens = generated[:, -self.max_seq_len:]
            else:
                input_tokens = generated
            
            output = self.forward(input_tokens)
            logits = output[:, -1, :]
            
            logits = logits / temperature
            
            if top_k is not None:
                top_k_vals = np.sort(logits, axis=-1)[:, -top_k:][:, 0:1]
                logits = np.where(logits < top_k_vals, float('-inf'), logits)
            
            probs = self._softmax(logits)
            
            if top_p is not None:
                sorted_indices = np.argsort(probs, axis=-1)[:, ::-1]
                sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)
                cumulative_probs = np.cumsum(sorted_probs, axis=-1)
                
                sorted_mask = cumulative_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].copy()
                sorted_mask[:, 0] = False
                
                for i in range(batch_size):
                    probs[i, sorted_indices[i][sorted_mask[i]]] = 0
                
                probs = probs / np.sum(probs, axis=-1, keepdims=True)
            
            next_tokens = np.array([
                np.random.choice(self.vocab_size, p=p) for p in probs
            ]).reshape(-1, 1)
            
            generated = np.concatenate([generated, next_tokens], axis=1)
        
        return generated
    
    def _softmax(self, x):
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def num_parameters(self):
        """
        Count total trainable parameters.
        
        Returns
        -------
        int
            Total number of parameters.
        """
        total = 0
        for layer in self.layers:
            for attr in ['W', 'b', 'E', 'Wq', 'Wk', 'Wv', 'Wo', 'gamma', 'beta',
                         'W1', 'W2', 'b1', 'b2']:
                if hasattr(layer, attr):
                    param = getattr(layer, attr)
                    if param is not None:
                        total += param.size
        return total
    
    def summary(self):
        """Print model summary."""
        print("=" * 70)
        print(f"Transformer Model")
        print(f"  Vocabulary Size: {self.vocab_size}")
        print(f"  Model Dimension: {self.d_model}")
        print(f"  Attention Heads: {self.n_heads}")
        print(f"  Layers: {self.n_layers}")
        print(f"  FF Dimension: {self.d_ff}")
        print(f"  Max Sequence Length: {self.max_seq_len}")
        print(f"  Causal: {self.causal}")
        print("=" * 70)
        print(f"{'Layer':<35} {'Params':<15}")
        print("=" * 70)
        
        total_params = 0
        
        embed_params = self.embedding.E.size if hasattr(self.embedding, 'E') else 0
        print(f"{'Embedding':<35} {embed_params:<15}")
        total_params += embed_params
        
        print(f"{'PositionalEncoding':<35} {0:<15}")
        
        for i, block in enumerate(self.transformer_blocks):
            block_params = 0
            for attr in ['W', 'b', 'Wq', 'Wk', 'Wv', 'Wo', 'gamma', 'beta',
                         'W1', 'W2', 'b1', 'b2']:
                if hasattr(block, attr):
                    param = getattr(block, attr)
                    if param is not None:
                        block_params += param.size
            
            for sub in ['attention', 'norm1', 'ff', 'norm2']:
                if hasattr(block, sub):
                    sublayer = getattr(block, sub)
                    for attr in ['W', 'b', 'Wq', 'Wk', 'Wv', 'Wo', 'gamma', 'beta',
                                 'W1', 'W2', 'b1', 'b2']:
                        if hasattr(sublayer, attr):
                            param = getattr(sublayer, attr)
                            if param is not None:
                                block_params += param.size
            
            print(f"{'TransformerBlock_' + str(i):<35} {block_params:<15}")
            total_params += block_params
        
        norm_params = (self.final_norm.gamma.size + self.final_norm.beta.size 
                       if hasattr(self.final_norm, 'gamma') else 0)
        print(f"{'LayerNorm (final)':<35} {norm_params:<15}")
        total_params += norm_params
        
        proj_params = self.output_proj.W.size + self.output_proj.b.size
        print(f"{'Linear (output)':<35} {proj_params:<15}")
        total_params += proj_params
        
        print("=" * 70)
        print(f"Total parameters: {total_params:,}")
        print("=" * 70)
    
    def save(self, filepath):
        """
        Save model weights.
        
        Parameters
        ----------
        filepath : str
            Path to save the model.
        """
        params = []
        for layer in self.layers:
            for attr in ['W', 'b', 'E', 'Wq', 'Wk', 'Wv', 'Wo', 'gamma', 'beta',
                         'W1', 'W2', 'b1', 'b2']:
                if hasattr(layer, attr):
                    param = getattr(layer, attr)
                    if param is not None:
                        params.append(param)
            
            for sub in ['attention', 'norm1', 'ff', 'norm2']:
                if hasattr(layer, sub):
                    sublayer = getattr(layer, sub)
                    for attr in ['W', 'b', 'Wq', 'Wk', 'Wv', 'Wo', 'gamma', 'beta',
                                 'W1', 'W2', 'b1', 'b2']:
                        if hasattr(sublayer, attr):
                            param = getattr(sublayer, attr)
                            if param is not None:
                                params.append(param)
        
        np.savez(filepath, *params)
    
    def load(self, filepath):
        """
        Load model weights.
        
        Parameters
        ----------
        filepath : str
            Path to load the model from.
        """
        data = np.load(filepath, allow_pickle=True)
        
        param_idx = 0
        for layer in self.layers:
            for attr in ['W', 'b', 'E', 'Wq', 'Wk', 'Wv', 'Wo', 'gamma', 'beta',
                         'W1', 'W2', 'b1', 'b2']:
                if hasattr(layer, attr) and f'arr_{param_idx}' in data.files:
                    setattr(layer, attr, data[f'arr_{param_idx}'])
                    param_idx += 1
            
            for sub in ['attention', 'norm1', 'ff', 'norm2']:
                if hasattr(layer, sub):
                    sublayer = getattr(layer, sub)
                    for attr in ['W', 'b', 'Wq', 'Wk', 'Wv', 'Wo', 'gamma', 'beta',
                                 'W1', 'W2', 'b1', 'b2']:
                        if hasattr(sublayer, attr) and f'arr_{param_idx}' in data.files:
                            setattr(sublayer, attr, data[f'arr_{param_idx}'])
                            param_idx += 1
    
    def __repr__(self):
        """String representation."""
        return (f"Transformer(vocab_size={self.vocab_size}, d_model={self.d_model}, "
                f"n_heads={self.n_heads}, n_layers={self.n_layers}, d_ff={self.d_ff})")
