from core.backend import (
    get_array_module, zeros, zeros_like, ones, random_randn, sqrt, dot, 
    tanh, exp, sum as xp_sum, stack
)


class LSTM:
    """
    Long Short-Term Memory Layer.
    
    Addresses the vanishing gradient problem in standard RNNs using
    a gating mechanism with forget, input, and output gates.
    
    Input shape: (batch_size, sequence_length, input_size)
    Output shape: (batch_size, sequence_length, hidden_size) or (batch_size, hidden_size)
    
    Supports both NumPy and CuPy backends transparently.
    """
    
    def __init__(self, input_size, hidden_size, return_sequences=True, initialization='xavier'):
        """
        Initialize the LSTM layer.
        
        Args:
            input_size: Size of the input features at each time step.
            hidden_size: Size of the hidden state and cell state.
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
            scale_ih = sqrt(2.0 / (self.input_size + self.hidden_size))
            scale_hh = sqrt(2.0 / (self.hidden_size + self.hidden_size))
        elif method == 'he':
            scale_ih = sqrt(2.0 / self.input_size)
            scale_hh = sqrt(2.0 / self.hidden_size)
        else:
            scale_ih = 0.01
            scale_hh = 0.01
            
        self.W_f = random_randn(self.input_size, self.hidden_size) * scale_ih
        self.U_f = random_randn(self.hidden_size, self.hidden_size) * scale_hh
        self.b_f = ones((1, self.hidden_size))
        
        self.W_i = random_randn(self.input_size, self.hidden_size) * scale_ih
        self.U_i = random_randn(self.hidden_size, self.hidden_size) * scale_hh
        self.b_i = zeros((1, self.hidden_size))
        
        self.W_c = random_randn(self.input_size, self.hidden_size) * scale_ih
        self.U_c = random_randn(self.hidden_size, self.hidden_size) * scale_hh
        self.b_c = zeros((1, self.hidden_size))
        
        self.W_o = random_randn(self.input_size, self.hidden_size) * scale_ih
        self.U_o = random_randn(self.hidden_size, self.hidden_size) * scale_hh
        self.b_o = zeros((1, self.hidden_size))
        
        self._init_gradients()
        
    def _init_gradients(self):
        """Initialize gradient accumulators."""
        self.dW_f = zeros_like(self.W_f)
        self.dU_f = zeros_like(self.U_f)
        self.db_f = zeros_like(self.b_f)
        
        self.dW_i = zeros_like(self.W_i)
        self.dU_i = zeros_like(self.U_i)
        self.db_i = zeros_like(self.b_i)
        
        self.dW_c = zeros_like(self.W_c)
        self.dU_c = zeros_like(self.U_c)
        self.db_c = zeros_like(self.b_c)
        
        self.dW_o = zeros_like(self.W_o)
        self.dU_o = zeros_like(self.U_o)
        self.db_o = zeros_like(self.b_o)
        
    def _sigmoid(self, x):
        """Numerically stable sigmoid."""
        xp = get_array_module(x)
        positive_mask = x >= 0
        negative_mask = ~positive_mask
        result = xp.zeros_like(x, dtype=xp.float64)
        result[positive_mask] = 1 / (1 + exp(-x[positive_mask]))
        exp_x = exp(x[negative_mask])
        result[negative_mask] = exp_x / (1 + exp_x)
        return result
        
    def forward(self, x, h_0=None, c_0=None):
        """
        Forward pass of the LSTM layer.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).
            h_0: Initial hidden state. Defaults to zeros.
            c_0: Initial cell state. Defaults to zeros.
            
        Returns:
            If return_sequences: Output tensor (batch_size, sequence_length, hidden_size).
            Else: Output tensor (batch_size, hidden_size).
        """
        xp = get_array_module(x)
        self.inputs = x
        batch_size, seq_length, _ = x.shape
        
        if h_0 is None:
            h_0 = xp.zeros((batch_size, self.hidden_size))
        if c_0 is None:
            c_0 = xp.zeros((batch_size, self.hidden_size))
            
        self.hidden_states = [h_0]
        self.cell_states = [c_0]
        
        self.forget_gates = []
        self.input_gates = []
        self.candidate_gates = []
        self.output_gates = []
        
        h_t = h_0
        c_t = c_0
        
        for t in range(seq_length):
            x_t = x[:, t, :]
            
            f_t = self._sigmoid(dot(x_t, self.W_f) + dot(h_t, self.U_f) + self.b_f)
            i_t = self._sigmoid(dot(x_t, self.W_i) + dot(h_t, self.U_i) + self.b_i)
            c_tilde = tanh(dot(x_t, self.W_c) + dot(h_t, self.U_c) + self.b_c)
            o_t = self._sigmoid(dot(x_t, self.W_o) + dot(h_t, self.U_o) + self.b_o)
            
            c_t = f_t * c_t + i_t * c_tilde
            h_t = o_t * tanh(c_t)
            
            self.forget_gates.append(f_t)
            self.input_gates.append(i_t)
            self.candidate_gates.append(c_tilde)
            self.output_gates.append(o_t)
            self.hidden_states.append(h_t)
            self.cell_states.append(c_t)
            
        self.hidden_states = self.hidden_states[1:]
        self.cell_states = self.cell_states[1:]
        
        if self.return_sequences:
            self.output = stack(self.hidden_states, axis=1)
        else:
            self.output = self.hidden_states[-1]
            
        return self.output
    
    def backward(self, gradient_output):
        """
        Backward pass of the LSTM layer using BPTT.
        
        Args:
            gradient_output: Gradient from the next layer.
            
        Returns:
            Gradient with respect to the input.
        """
        xp = get_array_module(self.inputs)
        batch_size, seq_length, _ = self.inputs.shape
        
        self._init_gradients()
        
        dx = xp.zeros_like(self.inputs)
        
        dh_next = xp.zeros((batch_size, self.hidden_size))
        dc_next = xp.zeros((batch_size, self.hidden_size))
        
        if not self.return_sequences:
            dh_seq = xp.zeros((batch_size, seq_length, self.hidden_size))
            dh_seq[:, -1, :] = gradient_output
            gradient_output = dh_seq
            
        for t in reversed(range(seq_length)):
            dh = gradient_output[:, t, :] + dh_next
            
            f_t = self.forget_gates[t]
            i_t = self.input_gates[t]
            c_tilde = self.candidate_gates[t]
            o_t = self.output_gates[t]
            c_t = self.cell_states[t]
            
            if t > 0:
                c_prev = self.cell_states[t - 1]
                h_prev = self.hidden_states[t - 1]
            else:
                c_prev = xp.zeros((batch_size, self.hidden_size))
                h_prev = xp.zeros((batch_size, self.hidden_size))
                
            tanh_c_t = tanh(c_t)
            
            do = dh * tanh_c_t
            dc = dh * o_t * (1 - tanh_c_t ** 2) + dc_next
            
            df = dc * c_prev
            di = dc * c_tilde
            dc_tilde = dc * i_t
            
            df_gate = df * f_t * (1 - f_t)
            di_gate = di * i_t * (1 - i_t)
            dc_gate = dc_tilde * (1 - c_tilde ** 2)
            do_gate = do * o_t * (1 - o_t)
            
            x_t = self.inputs[:, t, :]
            
            self.dW_f += dot(x_t.T, df_gate)
            self.dU_f += dot(h_prev.T, df_gate)
            self.db_f += xp_sum(df_gate, axis=0, keepdims=True)
            
            self.dW_i += dot(x_t.T, di_gate)
            self.dU_i += dot(h_prev.T, di_gate)
            self.db_i += xp_sum(di_gate, axis=0, keepdims=True)
            
            self.dW_c += dot(x_t.T, dc_gate)
            self.dU_c += dot(h_prev.T, dc_gate)
            self.db_c += xp_sum(dc_gate, axis=0, keepdims=True)
            
            self.dW_o += dot(x_t.T, do_gate)
            self.dU_o += dot(h_prev.T, do_gate)
            self.db_o += xp_sum(do_gate, axis=0, keepdims=True)
            
            dx[:, t, :] = (dot(df_gate, self.W_f.T) + dot(di_gate, self.W_i.T) +
                          dot(dc_gate, self.W_c.T) + dot(do_gate, self.W_o.T))
            
            dh_next = (dot(df_gate, self.U_f.T) + dot(di_gate, self.U_i.T) +
                      dot(dc_gate, self.U_c.T) + dot(do_gate, self.U_o.T))
            dc_next = dc * f_t
            
        return dx
    
    def get_weights(self):
        """Get all weights as a dictionary."""
        return {
            'W_f': self.W_f, 'U_f': self.U_f, 'b_f': self.b_f,
            'W_i': self.W_i, 'U_i': self.U_i, 'b_i': self.b_i,
            'W_c': self.W_c, 'U_c': self.U_c, 'b_c': self.b_c,
            'W_o': self.W_o, 'U_o': self.U_o, 'b_o': self.b_o
        }
    
    def set_weights(self, weights):
        """Set all weights from a dictionary."""
        self.W_f = weights['W_f']
        self.U_f = weights['U_f']
        self.b_f = weights['b_f']
        self.W_i = weights['W_i']
        self.U_i = weights['U_i']
        self.b_i = weights['b_i']
        self.W_c = weights['W_c']
        self.U_c = weights['U_c']
        self.b_c = weights['b_c']
        self.W_o = weights['W_o']
        self.U_o = weights['U_o']
        self.b_o = weights['b_o']
