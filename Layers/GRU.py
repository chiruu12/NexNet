import numpy as np


class GRU:
    """
    Gated Recurrent Unit Layer.
    
    A simpler alternative to LSTM with fewer parameters. Uses reset and
    update gates to control information flow through the sequence.
    
    Input shape: (batch_size, sequence_length, input_size)
    Output shape: (batch_size, sequence_length, hidden_size) or (batch_size, hidden_size)
    """
    
    def __init__(self, input_size, hidden_size, return_sequences=True, initialization='xavier'):
        """
        Initialize the GRU layer.
        
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
        """Initialize weights and biases for all gates."""
        if method == 'xavier':
            scale_ih = np.sqrt(2.0 / (self.input_size + self.hidden_size))
            scale_hh = np.sqrt(2.0 / (self.hidden_size + self.hidden_size))
        elif method == 'he':
            scale_ih = np.sqrt(2.0 / self.input_size)
            scale_hh = np.sqrt(2.0 / self.hidden_size)
        else:
            scale_ih = 0.01
            scale_hh = 0.01
            
        self.W_z = np.random.randn(self.input_size, self.hidden_size) * scale_ih
        self.U_z = np.random.randn(self.hidden_size, self.hidden_size) * scale_hh
        self.b_z = np.zeros((1, self.hidden_size))
        
        self.W_r = np.random.randn(self.input_size, self.hidden_size) * scale_ih
        self.U_r = np.random.randn(self.hidden_size, self.hidden_size) * scale_hh
        self.b_r = np.zeros((1, self.hidden_size))
        
        self.W_h = np.random.randn(self.input_size, self.hidden_size) * scale_ih
        self.U_h = np.random.randn(self.hidden_size, self.hidden_size) * scale_hh
        self.b_h = np.zeros((1, self.hidden_size))
        
        self._init_gradients()
        
    def _init_gradients(self):
        """Initialize gradient accumulators."""
        self.dW_z = np.zeros_like(self.W_z)
        self.dU_z = np.zeros_like(self.U_z)
        self.db_z = np.zeros_like(self.b_z)
        
        self.dW_r = np.zeros_like(self.W_r)
        self.dU_r = np.zeros_like(self.U_r)
        self.db_r = np.zeros_like(self.b_r)
        
        self.dW_h = np.zeros_like(self.W_h)
        self.dU_h = np.zeros_like(self.U_h)
        self.db_h = np.zeros_like(self.b_h)
        
    def _sigmoid(self, x):
        """Numerically stable sigmoid."""
        positive_mask = x >= 0
        negative_mask = ~positive_mask
        result = np.zeros_like(x, dtype=np.float64)
        result[positive_mask] = 1 / (1 + np.exp(-x[positive_mask]))
        exp_x = np.exp(x[negative_mask])
        result[negative_mask] = exp_x / (1 + exp_x)
        return result
        
    def forward(self, x, h_0=None):
        """
        Forward pass of the GRU layer.
        
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
        
        self.update_gates = []
        self.reset_gates = []
        self.candidate_states = []
        
        h_t = h_0
        
        for t in range(seq_length):
            x_t = x[:, t, :]
            
            z_t = self._sigmoid(np.dot(x_t, self.W_z) + np.dot(h_t, self.U_z) + self.b_z)
            r_t = self._sigmoid(np.dot(x_t, self.W_r) + np.dot(h_t, self.U_r) + self.b_r)
            
            h_tilde = np.tanh(np.dot(x_t, self.W_h) + np.dot(r_t * h_t, self.U_h) + self.b_h)
            
            h_t = (1 - z_t) * h_t + z_t * h_tilde
            
            self.update_gates.append(z_t)
            self.reset_gates.append(r_t)
            self.candidate_states.append(h_tilde)
            self.hidden_states.append(h_t)
            
        self.hidden_states = self.hidden_states[1:]
        
        if self.return_sequences:
            self.output = np.stack(self.hidden_states, axis=1)
        else:
            self.output = self.hidden_states[-1]
            
        return self.output
    
    def backward(self, gradient_output):
        """
        Backward pass of the GRU layer using BPTT.
        
        Args:
            gradient_output: Gradient from the next layer.
            
        Returns:
            Gradient with respect to the input.
        """
        batch_size, seq_length, _ = self.inputs.shape
        
        self._init_gradients()
        
        dx = np.zeros_like(self.inputs)
        
        dh_next = np.zeros((batch_size, self.hidden_size))
        
        if not self.return_sequences:
            dh_seq = np.zeros((batch_size, seq_length, self.hidden_size))
            dh_seq[:, -1, :] = gradient_output
            gradient_output = dh_seq
            
        for t in reversed(range(seq_length)):
            dh = gradient_output[:, t, :] + dh_next
            
            z_t = self.update_gates[t]
            r_t = self.reset_gates[t]
            h_tilde = self.candidate_states[t]
            h_t = self.hidden_states[t]
            
            if t > 0:
                h_prev = self.hidden_states[t - 1]
            else:
                h_prev = np.zeros((batch_size, self.hidden_size))
                
            dh_tilde = dh * z_t
            dz = dh * (h_tilde - h_prev)
            
            dh_tilde_pre = dh_tilde * (1 - h_tilde ** 2)
            dz_pre = dz * z_t * (1 - z_t)
            
            x_t = self.inputs[:, t, :]
            
            self.dW_h += np.dot(x_t.T, dh_tilde_pre)
            self.dU_h += np.dot((r_t * h_prev).T, dh_tilde_pre)
            self.db_h += np.sum(dh_tilde_pre, axis=0, keepdims=True)
            
            self.dW_z += np.dot(x_t.T, dz_pre)
            self.dU_z += np.dot(h_prev.T, dz_pre)
            self.db_z += np.sum(dz_pre, axis=0, keepdims=True)
            
            dr_h = np.dot(dh_tilde_pre, self.U_h.T)
            dr = dr_h * h_prev
            dr_pre = dr * r_t * (1 - r_t)
            
            self.dW_r += np.dot(x_t.T, dr_pre)
            self.dU_r += np.dot(h_prev.T, dr_pre)
            self.db_r += np.sum(dr_pre, axis=0, keepdims=True)
            
            dx[:, t, :] = (np.dot(dh_tilde_pre, self.W_h.T) + 
                          np.dot(dz_pre, self.W_z.T) + 
                          np.dot(dr_pre, self.W_r.T))
            
            dh_next = (dh * (1 - z_t) + 
                      dr_h * r_t + 
                      np.dot(dz_pre, self.U_z.T) + 
                      np.dot(dr_pre, self.U_r.T))
            
        return dx
    
    def get_weights(self):
        """Get all weights as a dictionary."""
        return {
            'W_z': self.W_z, 'U_z': self.U_z, 'b_z': self.b_z,
            'W_r': self.W_r, 'U_r': self.U_r, 'b_r': self.b_r,
            'W_h': self.W_h, 'U_h': self.U_h, 'b_h': self.b_h
        }
    
    def set_weights(self, weights):
        """Set all weights from a dictionary."""
        self.W_z = weights['W_z']
        self.U_z = weights['U_z']
        self.b_z = weights['b_z']
        self.W_r = weights['W_r']
        self.U_r = weights['U_r']
        self.b_r = weights['b_r']
        self.W_h = weights['W_h']
        self.U_h = weights['U_h']
        self.b_h = weights['b_h']
