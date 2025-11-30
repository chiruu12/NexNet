import numpy as np


class Module:
    """
    Base class for all neural network modules.
    
    Similar to PyTorch's nn.Module, this provides a unified interface
    for all layers and models in NexNet.
    
    Your models should subclass this class.
    
    Attributes
    ----------
    training : bool
        Whether the module is in training mode.
    _parameters : dict
        Dictionary of parameters (weights).
    _buffers : dict
        Dictionary of buffers (running stats, etc.).
    _modules : dict
        Dictionary of child modules.
        
    Examples
    --------
    >>> class MyModel(Module):
    ...     def __init__(self, in_features, out_features):
    ...         super().__init__()
    ...         self.linear = Linear(in_features, out_features)
    ...         self.relu = ReLu()
    ...     
    ...     def forward(self, x):
    ...         x = self.linear.forward(x)
    ...         x = self.relu.forward(x)
    ...         return x
    """
    
    def __init__(self):
        """Initialize the module."""
        self.training = True
        self._parameters = {}
        self._buffers = {}
        self._modules = {}
    
    def forward(self, *args, **kwargs):
        """
        Define the forward pass.
        
        This method should be overridden by all subclasses.
        
        Raises
        ------
        NotImplementedError
            If not overridden.
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def backward(self, *args, **kwargs):
        """
        Define the backward pass.
        
        This method should be overridden by all subclasses that need
        to compute gradients.
        """
        pass
    
    def __call__(self, *args, **kwargs):
        """Make module callable."""
        return self.forward(*args, **kwargs)
    
    def train(self, mode=True):
        """
        Set the module to training mode.
        
        Parameters
        ----------
        mode : bool, optional
            Whether to set training mode. Default is True.
            
        Returns
        -------
        self : Module
            Returns self for chaining.
        """
        self.training = mode
        for module in self._modules.values():
            if hasattr(module, 'train'):
                module.train(mode)
        return self
    
    def eval(self):
        """
        Set the module to evaluation mode.
        
        Equivalent to self.train(False).
        
        Returns
        -------
        self : Module
            Returns self for chaining.
        """
        return self.train(False)
    
    def parameters(self, recurse=True):
        """
        Return an iterator over module parameters.
        
        Parameters
        ----------
        recurse : bool, optional
            If True, also yields parameters of all submodules.
            Default is True.
            
        Yields
        ------
        Parameter
            Module parameters.
        """
        for name, param in self._parameters.items():
            yield param
        
        if recurse:
            for module in self._modules.values():
                if hasattr(module, 'parameters'):
                    yield from module.parameters(recurse=True)
    
    def named_parameters(self, prefix='', recurse=True):
        """
        Return an iterator over module parameters with their names.
        
        Parameters
        ----------
        prefix : str, optional
            Prefix to prepend to all parameter names.
        recurse : bool, optional
            If True, also yields parameters of all submodules.
            
        Yields
        ------
        tuple
            (name, parameter) pairs.
        """
        for name, param in self._parameters.items():
            yield prefix + name, param
        
        if recurse:
            for module_name, module in self._modules.items():
                if hasattr(module, 'named_parameters'):
                    submodule_prefix = prefix + module_name + '.'
                    yield from module.named_parameters(submodule_prefix, recurse=True)
    
    def buffers(self, recurse=True):
        """
        Return an iterator over module buffers.
        
        Parameters
        ----------
        recurse : bool, optional
            If True, also yields buffers of all submodules.
            
        Yields
        ------
        ndarray
            Module buffers.
        """
        for name, buf in self._buffers.items():
            yield buf
        
        if recurse:
            for module in self._modules.values():
                if hasattr(module, 'buffers'):
                    yield from module.buffers(recurse=True)
    
    def modules(self):
        """
        Return an iterator over all modules in the network.
        
        Yields
        ------
        Module
            Module in the network.
        """
        yield self
        for module in self._modules.values():
            if hasattr(module, 'modules'):
                yield from module.modules()
            else:
                yield module
    
    def named_modules(self, prefix=''):
        """
        Return an iterator over all modules with their names.
        
        Parameters
        ----------
        prefix : str, optional
            Prefix to prepend to all module names.
            
        Yields
        ------
        tuple
            (name, module) pairs.
        """
        yield prefix, self
        for name, module in self._modules.items():
            submodule_prefix = prefix + ('.' if prefix else '') + name
            if hasattr(module, 'named_modules'):
                yield from module.named_modules(submodule_prefix)
            else:
                yield submodule_prefix, module
    
    def register_parameter(self, name, param):
        """
        Add a parameter to the module.
        
        Parameters
        ----------
        name : str
            Name of the parameter.
        param : ndarray or None
            Parameter tensor or None.
        """
        self._parameters[name] = param
    
    def register_buffer(self, name, tensor):
        """
        Add a buffer to the module.
        
        Buffers are tensors that should be part of the module's state
        but are not considered parameters (e.g., running statistics).
        
        Parameters
        ----------
        name : str
            Name of the buffer.
        tensor : ndarray
            Buffer tensor.
        """
        self._buffers[name] = tensor
    
    def add_module(self, name, module):
        """
        Add a child module to the current module.
        
        Parameters
        ----------
        name : str
            Name of the child module.
        module : Module
            Child module to add.
        """
        self._modules[name] = module
    
    def zero_grad(self):
        """
        Set gradients of all parameters to zero.
        """
        for param in self.parameters():
            if hasattr(param, 'grad'):
                param.grad = None
    
    def num_parameters(self, only_trainable=True):
        """
        Count total number of parameters.
        
        Parameters
        ----------
        only_trainable : bool, optional
            If True, only count trainable parameters.
            Default is True.
            
        Returns
        -------
        int
            Total number of parameters.
        """
        total = 0
        for param in self.parameters():
            if param is not None:
                total += param.size
        return total
    
    def state_dict(self):
        """
        Return a dictionary containing the module's state.
        
        Returns
        -------
        dict
            Dictionary of state tensors.
        """
        state = {}
        
        for name, param in self._parameters.items():
            if param is not None:
                state[name] = param.copy()
        
        for name, buf in self._buffers.items():
            if buf is not None:
                state[name] = buf.copy()
        
        for module_name, module in self._modules.items():
            if hasattr(module, 'state_dict'):
                module_state = module.state_dict()
                for key, value in module_state.items():
                    state[f"{module_name}.{key}"] = value
        
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Load module state from a dictionary.
        
        Parameters
        ----------
        state_dict : dict
            Dictionary of state tensors.
        strict : bool, optional
            Whether to strictly enforce that the keys match.
            Default is True.
        """
        for name, param in self._parameters.items():
            if name in state_dict:
                self._parameters[name] = state_dict[name].copy()
            elif strict:
                raise KeyError(f"Missing parameter: {name}")
        
        for name, buf in self._buffers.items():
            if name in state_dict:
                self._buffers[name] = state_dict[name].copy()
        
        for module_name, module in self._modules.items():
            if hasattr(module, 'load_state_dict'):
                module_state = {
                    k.replace(f"{module_name}.", ""): v 
                    for k, v in state_dict.items() 
                    if k.startswith(f"{module_name}.")
                }
                module.load_state_dict(module_state, strict=strict)
    
    def save(self, filepath):
        """
        Save the module state to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the state.
        """
        state = self.state_dict()
        np.savez(filepath, **state)
    
    def load(self, filepath):
        """
        Load the module state from a file.
        
        Parameters
        ----------
        filepath : str
            Path to load the state from.
        """
        data = np.load(filepath, allow_pickle=True)
        state_dict = {key: data[key] for key in data.files}
        self.load_state_dict(state_dict, strict=False)
    
    def apply(self, fn):
        """
        Apply a function recursively to every submodule.
        
        Parameters
        ----------
        fn : callable
            Function to apply to each module.
            
        Returns
        -------
        self : Module
            Returns self for chaining.
        """
        for module in self._modules.values():
            if hasattr(module, 'apply'):
                module.apply(fn)
            fn(module)
        fn(self)
        return self
    
    def __repr__(self):
        """String representation."""
        return f"{self.__class__.__name__}()"


class Parameter:
    """
    A kind of Tensor that is to be considered a module parameter.
    
    Parameters are Tensor subclasses that have a special property
    when used with Modules - when they're assigned as Module attributes
    they are automatically added to the list of parameters.
    
    Parameters
    ----------
    data : ndarray
        Parameter tensor.
    requires_grad : bool, optional
        If True, gradients will be computed for this parameter.
        Default is True.
        
    Attributes
    ----------
    data : ndarray
        The parameter data.
    grad : ndarray or None
        The gradient of the parameter.
    requires_grad : bool
        Whether gradients are required.
        
    Examples
    --------
    >>> param = Parameter(np.random.randn(10, 5))
    >>> param.data.shape
    (10, 5)
    """
    
    def __init__(self, data, requires_grad=True):
        """
        Initialize the parameter.
        
        Parameters
        ----------
        data : ndarray
            Parameter data.
        requires_grad : bool, optional
            Whether to track gradients. Default is True.
        """
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
    
    @property
    def shape(self):
        """Return the shape of the parameter."""
        return self.data.shape
    
    @property
    def size(self):
        """Return the total number of elements."""
        return self.data.size
    
    @property
    def dtype(self):
        """Return the data type."""
        return self.data.dtype
    
    def zero_grad(self):
        """Set gradient to zero."""
        self.grad = np.zeros_like(self.data)
    
    def numpy(self):
        """Return the parameter as a numpy array."""
        return self.data
    
    def copy(self):
        """Return a copy of the parameter."""
        return Parameter(self.data.copy(), self.requires_grad)
    
    def __repr__(self):
        """String representation."""
        return f"Parameter(shape={self.shape}, requires_grad={self.requires_grad})"
    
    def __array__(self):
        """Support numpy array conversion."""
        return self.data


def init_weights(module, init_type='xavier_uniform', gain=1.0):
    """
    Initialize weights of a module.
    
    Parameters
    ----------
    module : Module or layer
        Module to initialize.
    init_type : str, optional
        Type of initialization. Options: 'xavier_uniform', 'xavier_normal',
        'he_uniform', 'he_normal', 'uniform', 'normal', 'zeros', 'ones'.
        Default is 'xavier_uniform'.
    gain : float, optional
        Scaling factor for Xavier initialization. Default is 1.0.
        
    Examples
    --------
    >>> model.apply(lambda m: init_weights(m, 'he_normal'))
    """
    if hasattr(module, 'W') and module.W is not None:
        fan_in = module.W.shape[0] if module.W.ndim > 1 else module.W.shape[0]
        fan_out = module.W.shape[1] if module.W.ndim > 1 else 1
        
        if init_type == 'xavier_uniform':
            limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
            module.W = np.random.uniform(-limit, limit, module.W.shape)
        elif init_type == 'xavier_normal':
            std = gain * np.sqrt(2.0 / (fan_in + fan_out))
            module.W = np.random.normal(0, std, module.W.shape)
        elif init_type == 'he_uniform':
            limit = np.sqrt(6.0 / fan_in)
            module.W = np.random.uniform(-limit, limit, module.W.shape)
        elif init_type == 'he_normal':
            std = np.sqrt(2.0 / fan_in)
            module.W = np.random.normal(0, std, module.W.shape)
        elif init_type == 'uniform':
            module.W = np.random.uniform(-1, 1, module.W.shape)
        elif init_type == 'normal':
            module.W = np.random.normal(0, 1, module.W.shape)
        elif init_type == 'zeros':
            module.W = np.zeros_like(module.W)
        elif init_type == 'ones':
            module.W = np.ones_like(module.W)
    
    if hasattr(module, 'b') and module.b is not None:
        module.b = np.zeros_like(module.b)
