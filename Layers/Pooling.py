import numpy as np


class MaxPool2D:
    """
    2D Max Pooling Layer.
    
    Downsamples input by taking the maximum value in each pooling window.
    Reduces spatial dimensions while retaining important features.
    """
    
    def __init__(self, pool_size=2, stride=None):
        """
        Initialize the MaxPool2D layer.
        
        Args:
            pool_size: Size of the pooling window (int or tuple).
            stride: Stride of the pooling operation. Defaults to pool_size.
        """
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size
            
        if stride is None:
            self.stride = self.pool_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
    def forward(self, x):
        """
        Forward pass of the MaxPool2D layer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            Output tensor with reduced spatial dimensions.
        """
        self.input_shape = x.shape
        batch_size, channels, h, w = x.shape
        ph, pw = self.pool_size
        sh, sw = self.stride
        
        out_h = (h - ph) // sh + 1
        out_w = (w - pw) // sw + 1
        
        self.output = np.zeros((batch_size, channels, out_h, out_w))
        self.max_indices = np.zeros((batch_size, channels, out_h, out_w, 2), dtype=int)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * sh
                h_end = h_start + ph
                w_start = j * sw
                w_end = w_start + pw
                
                window = x[:, :, h_start:h_end, w_start:w_end]
                
                window_reshaped = window.reshape(batch_size, channels, -1)
                max_idx = np.argmax(window_reshaped, axis=2)
                
                max_h = max_idx // pw
                max_w = max_idx % pw
                
                self.max_indices[:, :, i, j, 0] = h_start + max_h
                self.max_indices[:, :, i, j, 1] = w_start + max_w
                
                self.output[:, :, i, j] = np.max(window_reshaped, axis=2)
                
        return self.output
    
    def backward(self, gradient_output):
        """
        Backward pass of the MaxPool2D layer.
        
        Args:
            gradient_output: Gradient from the next layer.
            
        Returns:
            Gradient with respect to the input.
        """
        batch_size, channels, out_h, out_w = gradient_output.shape
        dx = np.zeros(self.input_shape)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_idx = self.max_indices[b, c, i, j, 0]
                        w_idx = self.max_indices[b, c, i, j, 1]
                        dx[b, c, h_idx, w_idx] += gradient_output[b, c, i, j]
                        
        return dx


class AvgPool2D:
    """
    2D Average Pooling Layer.
    
    Downsamples input by taking the average value in each pooling window.
    Provides smoother downsampling compared to max pooling.
    """
    
    def __init__(self, pool_size=2, stride=None):
        """
        Initialize the AvgPool2D layer.
        
        Args:
            pool_size: Size of the pooling window (int or tuple).
            stride: Stride of the pooling operation. Defaults to pool_size.
        """
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size
            
        if stride is None:
            self.stride = self.pool_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
    def forward(self, x):
        """
        Forward pass of the AvgPool2D layer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            Output tensor with reduced spatial dimensions.
        """
        self.input_shape = x.shape
        batch_size, channels, h, w = x.shape
        ph, pw = self.pool_size
        sh, sw = self.stride
        
        out_h = (h - ph) // sh + 1
        out_w = (w - pw) // sw + 1
        
        self.output = np.zeros((batch_size, channels, out_h, out_w))
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * sh
                h_end = h_start + ph
                w_start = j * sw
                w_end = w_start + pw
                
                window = x[:, :, h_start:h_end, w_start:w_end]
                self.output[:, :, i, j] = np.mean(window, axis=(2, 3))
                
        return self.output
    
    def backward(self, gradient_output):
        """
        Backward pass of the AvgPool2D layer.
        
        Args:
            gradient_output: Gradient from the next layer.
            
        Returns:
            Gradient with respect to the input.
        """
        batch_size, channels, out_h, out_w = gradient_output.shape
        ph, pw = self.pool_size
        sh, sw = self.stride
        
        dx = np.zeros(self.input_shape)
        pool_area = ph * pw
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * sh
                h_end = h_start + ph
                w_start = j * sw
                w_end = w_start + pw
                
                grad = gradient_output[:, :, i, j][:, :, np.newaxis, np.newaxis]
                dx[:, :, h_start:h_end, w_start:w_end] += grad / pool_area
                
        return dx
