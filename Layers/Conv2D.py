from core.backend import get_array_module, zeros, zeros_like, random_randn, sqrt, pad, dot


class Conv2D:
    """
    2D Convolutional Layer.
    
    Performs convolution operation on 4D input tensors (batch, channels, height, width).
    Supports configurable kernel size, stride, padding, and multiple filters.
    Supports both CPU (NumPy) and GPU (CuPy) backends.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialization='he'):
        """
        Initialize the Conv2D layer.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (number of filters).
            kernel_size: Size of the convolutional kernel (int or tuple).
            stride: Stride of the convolution (int or tuple).
            padding: Zero-padding added to both sides of the input.
            initialization: Weight initialization method ('he', 'xavier', 'random').
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        self.padding = padding
        
        self._initialize_weights(initialization)
        
    def _initialize_weights(self, method):
        """Initialize weights and biases."""
        xp = get_array_module()
        kh, kw = self.kernel_size
        
        if method == 'he':
            scale = sqrt(2.0 / (self.in_channels * kh * kw))
        elif method == 'xavier':
            scale = sqrt(2.0 / (self.in_channels * kh * kw + self.out_channels * kh * kw))
        else:
            scale = 0.01
            
        self.W = random_randn(self.out_channels, self.in_channels, kh, kw) * scale
        self.b = zeros((self.out_channels, 1))
        
        self.dW = zeros_like(self.W)
        self.db = zeros_like(self.b)
        
    def _pad_input(self, x):
        """Add zero padding to input."""
        if self.padding == 0:
            return x
        return pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
    
    def _get_output_shape(self, input_shape):
        """Calculate output dimensions."""
        batch_size, _, h, w = input_shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        
        out_h = (h + 2 * self.padding - kh) // sh + 1
        out_w = (w + 2 * self.padding - kw) // sw + 1
        
        return batch_size, self.out_channels, out_h, out_w
    
    def _im2col(self, x, kh, kw, stride):
        """
        Transform input into column matrix for efficient convolution.
        
        Converts image patches into columns for matrix multiplication.
        """
        xp = get_array_module(x)
        batch_size, channels, h, w = x.shape
        sh, sw = stride
        
        out_h = (h - kh) // sh + 1
        out_w = (w - kw) // sw + 1
        
        col = xp.zeros((batch_size, channels, kh, kw, out_h, out_w))
        
        for i in range(kh):
            i_max = i + sh * out_h
            for j in range(kw):
                j_max = j + sw * out_w
                col[:, :, i, j, :, :] = x[:, :, i:i_max:sh, j:j_max:sw]
                
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(batch_size * out_h * out_w, -1)
        return col
    
    def _col2im(self, col, x_shape, kh, kw, stride):
        """
        Transform column matrix back to image format.
        
        Inverse operation of im2col for backpropagation.
        """
        xp = get_array_module(col)
        batch_size, channels, h, w = x_shape
        sh, sw = stride
        
        out_h = (h - kh) // sh + 1
        out_w = (w - kw) // sw + 1
        
        col = col.reshape(batch_size, out_h, out_w, channels, kh, kw).transpose(0, 3, 4, 5, 1, 2)
        
        x = xp.zeros(x_shape)
        
        for i in range(kh):
            i_max = i + sh * out_h
            for j in range(kw):
                j_max = j + sw * out_w
                x[:, :, i:i_max:sh, j:j_max:sw] += col[:, :, i, j, :, :]
                
        return x
        
    def forward(self, x):
        """
        Forward pass of the Conv2D layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            Output tensor of shape (batch_size, out_channels, out_height, out_width).
        """
        xp = get_array_module(x)
        self.input_shape = x.shape
        self.x_padded = self._pad_input(x)
        
        batch_size, out_channels, out_h, out_w = self._get_output_shape(x.shape)
        kh, kw = self.kernel_size
        
        self.col = self._im2col(self.x_padded, kh, kw, self.stride)
        
        W_col = self.W.reshape(self.out_channels, -1)
        
        output = dot(self.col, W_col.T) + self.b.T
        
        output = output.reshape(batch_size, out_h, out_w, self.out_channels)
        self.output = output.transpose(0, 3, 1, 2)
        
        return self.output
    
    def backward(self, gradient_output):
        """
        Backward pass of the Conv2D layer.
        
        Args:
            gradient_output: Gradient from the next layer.
            
        Returns:
            Gradient with respect to the input.
        """
        xp = get_array_module(gradient_output)
        batch_size = gradient_output.shape[0]
        kh, kw = self.kernel_size
        
        gradient_output_reshaped = gradient_output.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        
        W_col = self.W.reshape(self.out_channels, -1)
        
        self.dW = dot(gradient_output_reshaped.T, self.col).reshape(self.W.shape)
        self.db = xp.sum(gradient_output_reshaped, axis=0).reshape(self.b.shape)
        
        dcol = dot(gradient_output_reshaped, W_col)
        
        dx_padded = self._col2im(dcol, self.x_padded.shape, kh, kw, self.stride)
        
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded
            
        return dx
