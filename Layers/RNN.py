import numpy as np


class RNN:
    """
    Vanilla Recurrent Neural Network Layer.
    
    Processes sequential data by maintaining a hidden state that is
    updated at each time step. Suitable for simple sequence tasks.
    
    Input shape: (batch_size, sequence_length, input_size)
    Output shape: (batch_size, sequence_length, hidden_size) or (batch_size, hidden_size)
    """
    
    def __init__(self, input_size, hidden_size, return_sequences=True, initialization='xavier'):
        """
        Initialize the RNN layer.
        
        Args:
            input_size: Size of the input features at each time step.
            hidden_size: Size of the hidden state.
            return_sequences: If True, return outputs for all time steps.
                            If False, return only the last output.
            initialization: Weight initialization method ('xavier', 'he', 'random').
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        
        self._initialize_weights(initialization)
        
    def _initialize_weights(self, method):
        """Initialize weights and biases."""
        if method == 'xavier':
            scale_ih = np.sqrt(2.0 / (self.input_size + self.hidden_size))
            scale_hh = np.sqrt(2.0 / (self.hidden_size + self.hidden_size))
        elif method == 'he':
            scale_ih = np.sqrt(2.0 / self.input_size)
            scale_hh = np.sqrt(2.0 / self.hidden_size)
        else:
            scale_ih = 0.01
            scale_hh = 0.01
            
        self.W_ih = np.random.randn(self.input_size, self.hidden_size) * scale_ih
        self.W_hh = np.random.randn(self.hidden_size, self.hidden_size) * scale_hh
        self.b_h = np.zeros((1, self.hidden_size))
        
        self.dW_ih = np.zeros_like(self.W_ih)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.db_h = np.zeros_like(self.b_h)
        
    def forward(self, x, h_0=None):
        """
        Forward pass of the RNN layer.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).
            h_0: Initial hidden state. Defaults to zeros.
            
        Returns:
            If return_sequences: Output tensor (batch_size, sequence_length, hidden_size).
            Else: Output tensor (batch_size, hidden_size).
        """
        self.inputs = x
        batch_size, seq_length, _ = x.shape
        
        if h_0 is None:
            h_0 = np.zeros((batch_size, self.hidden_size))
            
        self.hidden_states = [h_0]
        self.pre_activations = []
        
        h_t = h_0
        for t in range(seq_length):
            x_t = x[:, t, :]
            
            pre_act = np.dot(x_t, self.W_ih) + np.dot(h_t, self.W_hh) + self.b_h
            self.pre_activations.append(pre_act)
            
            h_t = np.tanh(pre_act)
            self.hidden_states.append(h_t)
            
        self.hidden_states = self.hidden_states[1:]
        
        if self.return_sequences:
            self.output = np.stack(self.hidden_states, axis=1)
        else:
            self.output = self.hidden_states[-1]
            
        return self.output
    
    def backward(self, gradient_output):
        """
        Backward pass of the RNN layer using Backpropagation Through Time (BPTT).
        
        Args:
            gradient_output: Gradient from the next layer.
            
        Returns:
            Gradient with respect to the input.
        """
        batch_size, seq_length, _ = self.inputs.shape
        
        self.dW_ih = np.zeros_like(self.W_ih)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.db_h = np.zeros_like(self.b_h)
        
        dx = np.zeros_like(self.inputs)
        
        if self.return_sequences:
            dh_next = np.zeros((batch_size, self.hidden_size))
        else:
            dh_next = gradient_output
            gradient_output = np.zeros((batch_size, seq_length, self.hidden_size))
            gradient_output[:, -1, :] = dh_next
            dh_next = np.zeros((batch_size, self.hidden_size))
            
        for t in reversed(range(seq_length)):
            if self.return_sequences:
                dh = gradient_output[:, t, :] + dh_next
            else:
                dh = dh_next if t < seq_length - 1 else gradient_output[:, t, :] + dh_next
                
            h_t = self.hidden_states[t]
            dtanh = dh * (1 - h_t ** 2)
            
            self.db_h += np.sum(dtanh, axis=0, keepdims=True)
            self.dW_ih += np.dot(self.inputs[:, t, :].T, dtanh)
            
            if t > 0:
                h_prev = self.hidden_states[t - 1]
            else:
                h_prev = np.zeros((batch_size, self.hidden_size))
                
            self.dW_hh += np.dot(h_prev.T, dtanh)
            
            dx[:, t, :] = np.dot(dtanh, self.W_ih.T)
            
            dh_next = np.dot(dtanh, self.W_hh.T)
            
        return dx
