"""
Backend Abstraction Module.

Provides a unified interface for NumPy and CuPy operations.
This allows seamless switching between CPU (NumPy) and GPU (CuPy)
without modifying the rest of the codebase.

Usage:
    from core.backend import get_array_module, to_cpu, to_gpu, set_backend
    
    # Get the current backend (np or cp)
    xp = get_array_module()
    
    # Use xp instead of np everywhere
    x = xp.array([1, 2, 3])
    y = xp.zeros((10, 10))
    
    # Switch backends
    set_backend('gpu')  # or 'cuda', 'cupy'
    set_backend('cpu')  # or 'numpy'
    
    # Transfer data between devices
    gpu_array = to_gpu(numpy_array)
    cpu_array = to_cpu(cupy_array)
"""

import numpy as np

_BACKEND = 'numpy'
_CUPY_AVAILABLE = False

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None


def is_cupy_available():
    """Check if CuPy is available."""
    return _CUPY_AVAILABLE


def get_backend():
    """Get the current backend name."""
    return _BACKEND


def set_backend(backend):
    """
    Set the computation backend.
    
    Args:
        backend: 'numpy', 'cpu', 'cupy', 'gpu', or 'cuda'
    
    Raises:
        ValueError: If backend is invalid.
        RuntimeError: If CuPy is requested but not available.
    """
    global _BACKEND
    
    backend = backend.lower()
    
    if backend in ('numpy', 'cpu'):
        _BACKEND = 'numpy'
    elif backend in ('cupy', 'gpu', 'cuda'):
        if not _CUPY_AVAILABLE:
            raise RuntimeError(
                "CuPy is not installed. Install it with:\n"
                "  pip install cupy-cuda11x  # For CUDA 11.x\n"
                "  pip install cupy-cuda12x  # For CUDA 12.x\n"
                "Or see: https://docs.cupy.dev/en/stable/install.html"
            )
        _BACKEND = 'cupy'
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'numpy', 'cpu', 'cupy', 'gpu', or 'cuda'.")


def get_array_module(x=None):
    """
    Get the array module (numpy or cupy) for computation.
    
    If an array is provided, returns the module that owns that array.
    Otherwise, returns the currently configured backend.
    
    Args:
        x: Optional array to detect backend from.
        
    Returns:
        numpy or cupy module.
    """
    if x is not None:
        if _CUPY_AVAILABLE and hasattr(cp, 'get_array_module'):
            return cp.get_array_module(x)
        if _CUPY_AVAILABLE and isinstance(x, cp.ndarray):
            return cp
        return np
    
    if _BACKEND == 'cupy':
        return cp
    return np


def to_cpu(x):
    """
    Transfer array to CPU (NumPy).
    
    Args:
        x: Input array (numpy or cupy).
        
    Returns:
        NumPy array.
    """
    if x is None:
        return None
    if _CUPY_AVAILABLE and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def to_gpu(x):
    """
    Transfer array to GPU (CuPy).
    
    Args:
        x: Input array (numpy or cupy).
        
    Returns:
        CuPy array.
        
    Raises:
        RuntimeError: If CuPy is not available.
    """
    if x is None:
        return None
    if not _CUPY_AVAILABLE:
        raise RuntimeError("CuPy is not available. Cannot transfer to GPU.")
    if isinstance(x, cp.ndarray):
        return x
    return cp.asarray(x)


def to_device(x, device='cpu'):
    """
    Transfer array to specified device.
    
    Args:
        x: Input array.
        device: 'cpu' or 'gpu'/'cuda'.
        
    Returns:
        Array on the specified device.
    """
    if device in ('gpu', 'cuda', 'cupy'):
        return to_gpu(x)
    return to_cpu(x)


def zeros(shape, dtype=None):
    """Create array of zeros using current backend."""
    xp = get_array_module()
    return xp.zeros(shape, dtype=dtype)


def ones(shape, dtype=None):
    """Create array of ones using current backend."""
    xp = get_array_module()
    return xp.ones(shape, dtype=dtype)


def zeros_like(x):
    """Create array of zeros with same shape and type."""
    xp = get_array_module(x)
    return xp.zeros_like(x)


def ones_like(x):
    """Create array of ones with same shape and type."""
    xp = get_array_module(x)
    return xp.ones_like(x)


def array(x, dtype=None):
    """Create array using current backend."""
    xp = get_array_module()
    return xp.array(x, dtype=dtype)


def asarray(x, dtype=None):
    """Convert to array using current backend."""
    xp = get_array_module()
    return xp.asarray(x, dtype=dtype)


def random_randn(*shape):
    """Generate random normal values using current backend."""
    xp = get_array_module()
    return xp.random.randn(*shape)


def random_rand(*shape):
    """Generate random uniform values [0, 1) using current backend."""
    xp = get_array_module()
    return xp.random.rand(*shape)


def random_uniform(low=0.0, high=1.0, size=None):
    """Generate random uniform values in [low, high) using current backend."""
    xp = get_array_module()
    return xp.random.uniform(low, high, size=size)


def random_normal(loc=0.0, scale=1.0, size=None):
    """Generate random normal values using current backend."""
    xp = get_array_module()
    return xp.random.normal(loc, scale, size=size)


def random_randint(low, high=None, size=None):
    """Generate random integers using current backend."""
    xp = get_array_module()
    return xp.random.randint(low, high, size=size)


def random_permutation(n):
    """Generate random permutation using current backend."""
    xp = get_array_module()
    return xp.random.permutation(n)


def random_choice(a, size=None, replace=True, p=None):
    """Random choice using current backend."""
    xp = get_array_module()
    return xp.random.choice(a, size=size, replace=replace, p=p)


def seed(s):
    """Set random seed for current backend."""
    xp = get_array_module()
    xp.random.seed(s)
    np.random.seed(s)


def arange(start, stop=None, step=1, dtype=None):
    """Create range array using current backend."""
    xp = get_array_module()
    if stop is None:
        return xp.arange(start, dtype=dtype)
    return xp.arange(start, stop, step, dtype=dtype)


def linspace(start, stop, num=50, dtype=None):
    """Create linearly spaced array using current backend."""
    xp = get_array_module()
    return xp.linspace(start, stop, num, dtype=dtype)


def eye(n, m=None, dtype=None):
    """Create identity matrix using current backend."""
    xp = get_array_module()
    return xp.eye(n, m, dtype=dtype)


def concatenate(arrays, axis=0):
    """Concatenate arrays using appropriate backend."""
    if len(arrays) == 0:
        return array([])
    xp = get_array_module(arrays[0])
    return xp.concatenate(arrays, axis=axis)


def stack(arrays, axis=0):
    """Stack arrays using appropriate backend."""
    if len(arrays) == 0:
        return array([])
    xp = get_array_module(arrays[0])
    return xp.stack(arrays, axis=axis)


def vstack(arrays):
    """Vertically stack arrays."""
    if len(arrays) == 0:
        return array([])
    xp = get_array_module(arrays[0])
    return xp.vstack(arrays)


def hstack(arrays):
    """Horizontally stack arrays."""
    if len(arrays) == 0:
        return array([])
    xp = get_array_module(arrays[0])
    return xp.hstack(arrays)


def split(x, indices_or_sections, axis=0):
    """Split array."""
    xp = get_array_module(x)
    return xp.split(x, indices_or_sections, axis=axis)


def clip(x, a_min, a_max):
    """Clip array values."""
    xp = get_array_module(x)
    return xp.clip(x, a_min, a_max)


def where(condition, x=None, y=None):
    """Conditional selection."""
    xp = get_array_module(condition)
    if x is None and y is None:
        return xp.where(condition)
    return xp.where(condition, x, y)


def maximum(x1, x2):
    """Element-wise maximum."""
    xp = get_array_module(x1)
    return xp.maximum(x1, x2)


def minimum(x1, x2):
    """Element-wise minimum."""
    xp = get_array_module(x1)
    return xp.minimum(x1, x2)


def abs(x):
    """Absolute value."""
    xp = get_array_module(x)
    return xp.abs(x)


def sign(x):
    """Sign function."""
    xp = get_array_module(x)
    return xp.sign(x)


def sqrt(x):
    """Square root."""
    xp = get_array_module(x)
    return xp.sqrt(x)


def square(x):
    """Square."""
    xp = get_array_module(x)
    return xp.square(x)


def floor(x):
    """Floor."""
    xp = get_array_module(x)
    return xp.floor(x)


def ceil(x):
    """Ceiling."""
    xp = get_array_module(x)
    return xp.ceil(x)


def round(x, decimals=0):
    """Round to specified decimals."""
    xp = get_array_module(x)
    return xp.round(x, decimals)


def power(x, p):
    """Power."""
    xp = get_array_module(x)
    return xp.power(x, p)


def exp(x):
    """Exponential."""
    xp = get_array_module(x)
    return xp.exp(x)


def log(x):
    """Natural logarithm."""
    xp = get_array_module(x)
    return xp.log(x)


def log2(x):
    """Base-2 logarithm."""
    xp = get_array_module(x)
    return xp.log2(x)


def log10(x):
    """Base-10 logarithm."""
    xp = get_array_module(x)
    return xp.log10(x)


def sin(x):
    """Sine."""
    xp = get_array_module(x)
    return xp.sin(x)


def cos(x):
    """Cosine."""
    xp = get_array_module(x)
    return xp.cos(x)


def tan(x):
    """Tangent."""
    xp = get_array_module(x)
    return xp.tan(x)


def tanh(x):
    """Hyperbolic tangent."""
    xp = get_array_module(x)
    return xp.tanh(x)


def sinh(x):
    """Hyperbolic sine."""
    xp = get_array_module(x)
    return xp.sinh(x)


def cosh(x):
    """Hyperbolic cosine."""
    xp = get_array_module(x)
    return xp.cosh(x)


def sum(x, axis=None, keepdims=False):
    """Sum of array elements."""
    xp = get_array_module(x)
    return xp.sum(x, axis=axis, keepdims=keepdims)


def mean(x, axis=None, keepdims=False):
    """Mean of array elements."""
    xp = get_array_module(x)
    return xp.mean(x, axis=axis, keepdims=keepdims)


def std(x, axis=None, keepdims=False, ddof=0):
    """Standard deviation."""
    xp = get_array_module(x)
    return xp.std(x, axis=axis, keepdims=keepdims, ddof=ddof)


def var(x, axis=None, keepdims=False, ddof=0):
    """Variance."""
    xp = get_array_module(x)
    return xp.var(x, axis=axis, keepdims=keepdims, ddof=ddof)


def max(x, axis=None, keepdims=False):
    """Maximum value."""
    xp = get_array_module(x)
    return xp.max(x, axis=axis, keepdims=keepdims)


def min(x, axis=None, keepdims=False):
    """Minimum value."""
    xp = get_array_module(x)
    return xp.min(x, axis=axis, keepdims=keepdims)


def argmax(x, axis=None):
    """Index of maximum value."""
    xp = get_array_module(x)
    return xp.argmax(x, axis=axis)


def argmin(x, axis=None):
    """Index of minimum value."""
    xp = get_array_module(x)
    return xp.argmin(x, axis=axis)


def argsort(x, axis=-1):
    """Indices that would sort array."""
    xp = get_array_module(x)
    return xp.argsort(x, axis=axis)


def sort(x, axis=-1):
    """Sort array."""
    xp = get_array_module(x)
    return xp.sort(x, axis=axis)


def prod(x, axis=None, keepdims=False):
    """Product of array elements."""
    xp = get_array_module(x)
    return xp.prod(x, axis=axis, keepdims=keepdims)


def cumsum(x, axis=None):
    """Cumulative sum."""
    xp = get_array_module(x)
    return xp.cumsum(x, axis=axis)


def dot(a, b):
    """Dot product."""
    xp = get_array_module(a)
    return xp.dot(a, b)


def matmul(a, b):
    """Matrix multiplication."""
    xp = get_array_module(a)
    return xp.matmul(a, b)


def tensordot(a, b, axes):
    """Tensor dot product."""
    xp = get_array_module(a)
    return xp.tensordot(a, b, axes=axes)


def einsum(subscripts, *operands):
    """Einstein summation."""
    xp = get_array_module(operands[0])
    return xp.einsum(subscripts, *operands)


def transpose(x, axes=None):
    """Transpose array."""
    xp = get_array_module(x)
    return xp.transpose(x, axes=axes)


def reshape(x, shape):
    """Reshape array."""
    xp = get_array_module(x)
    return xp.reshape(x, shape)


def expand_dims(x, axis):
    """Expand dimensions."""
    xp = get_array_module(x)
    return xp.expand_dims(x, axis=axis)


def squeeze(x, axis=None):
    """Remove single-dimensional entries."""
    xp = get_array_module(x)
    return xp.squeeze(x, axis=axis)


def swapaxes(x, axis1, axis2):
    """Swap axes."""
    xp = get_array_module(x)
    return xp.swapaxes(x, axis1, axis2)


def moveaxis(x, source, destination):
    """Move axes."""
    xp = get_array_module(x)
    return xp.moveaxis(x, source, destination)


def broadcast_to(x, shape):
    """Broadcast to shape."""
    xp = get_array_module(x)
    return xp.broadcast_to(x, shape)


def tile(x, reps):
    """Tile array."""
    xp = get_array_module(x)
    return xp.tile(x, reps)


def repeat(x, repeats, axis=None):
    """Repeat elements."""
    xp = get_array_module(x)
    return xp.repeat(x, repeats, axis=axis)


def flip(x, axis=None):
    """Flip array."""
    xp = get_array_module(x)
    return xp.flip(x, axis=axis)


def roll(x, shift, axis=None):
    """Roll array elements."""
    xp = get_array_module(x)
    return xp.roll(x, shift, axis=axis)


def pad(x, pad_width, mode='constant', constant_values=0):
    """Pad array."""
    xp = get_array_module(x)
    if mode == 'constant':
        return xp.pad(x, pad_width, mode=mode, constant_values=constant_values)
    return xp.pad(x, pad_width, mode=mode)


def allclose(a, b, rtol=1e-5, atol=1e-8):
    """Check if arrays are close."""
    xp = get_array_module(a)
    return xp.allclose(a, b, rtol=rtol, atol=atol)


def isnan(x):
    """Check for NaN."""
    xp = get_array_module(x)
    return xp.isnan(x)


def isinf(x):
    """Check for Inf."""
    xp = get_array_module(x)
    return xp.isinf(x)


def isfinite(x):
    """Check for finite values."""
    xp = get_array_module(x)
    return xp.isfinite(x)


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    """Replace NaN and Inf."""
    xp = get_array_module(x)
    return xp.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def take_along_axis(arr, indices, axis):
    """Take values along axis."""
    xp = get_array_module(arr)
    return xp.take_along_axis(arr, indices, axis)


def put_along_axis(arr, indices, values, axis):
    """Put values along axis."""
    xp = get_array_module(arr)
    xp.put_along_axis(arr, indices, values, axis)


def unique(x, return_counts=False):
    """Find unique elements."""
    xp = get_array_module(x)
    return xp.unique(x, return_counts=return_counts)


def bincount(x, weights=None, minlength=0):
    """Count occurrences."""
    xp = get_array_module(x)
    return xp.bincount(x, weights=weights, minlength=minlength)


def norm(x, ord=None, axis=None, keepdims=False):
    """Compute norm."""
    xp = get_array_module(x)
    if hasattr(xp, 'linalg'):
        return xp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
    if ord is None or ord == 2:
        return xp.sqrt(xp.sum(x ** 2, axis=axis, keepdims=keepdims))
    elif ord == 1:
        return xp.sum(xp.abs(x), axis=axis, keepdims=keepdims)
    elif ord == float('inf'):
        return xp.max(xp.abs(x), axis=axis, keepdims=keepdims)
    else:
        return xp.power(xp.sum(xp.power(xp.abs(x), ord), axis=axis, keepdims=keepdims), 1.0/ord)


def triu(x, k=0):
    """Upper triangle of array."""
    xp = get_array_module(x)
    return xp.triu(x, k=k)


def tril(x, k=0):
    """Lower triangle of array."""
    xp = get_array_module(x)
    return xp.tril(x, k=k)


def diag(x, k=0):
    """Extract diagonal or create diagonal array."""
    xp = get_array_module(x)
    return xp.diag(x, k=k)


def trace(x, offset=0, axis1=0, axis2=1):
    """Sum along diagonal."""
    xp = get_array_module(x)
    return xp.trace(x, offset=offset, axis1=axis1, axis2=axis2)


def diagflat(x, k=0):
    """Create diagonal array from flat input."""
    xp = get_array_module(x)
    return xp.diagflat(x, k=k)


def outer(a, b):
    """Outer product of two arrays."""
    xp = get_array_module(a)
    return xp.outer(a, b)


def erf(x):
    """
    Error function.
    
    Falls back to approximation if scipy not available on GPU.
    """
    xp = get_array_module(x)
    
    if xp == np:
        try:
            from scipy.special import erf as scipy_erf
            return scipy_erf(x)
        except ImportError:
            pass
    
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    sign = xp.sign(x)
    x = xp.abs(x)
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * xp.exp(-x * x)
    
    return sign * y


class DeviceContext:
    """
    Context manager for temporarily switching backends.
    
    Usage:
        with DeviceContext('gpu'):
            # GPU operations here
            x = xp.zeros((100, 100))
        # Back to previous backend
    """
    
    def __init__(self, backend):
        self.backend = backend
        self.previous_backend = None
        
    def __enter__(self):
        self.previous_backend = get_backend()
        set_backend(self.backend)
        return get_array_module()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        set_backend(self.previous_backend)
        return False


def use_gpu():
    """Context manager for GPU operations."""
    return DeviceContext('gpu')


def use_cpu():
    """Context manager for CPU operations."""
    return DeviceContext('cpu')


def get_device_info():
    """Get information about available devices."""
    info = {
        'backend': _BACKEND,
        'cupy_available': _CUPY_AVAILABLE,
        'numpy_version': np.__version__,
    }
    
    if _CUPY_AVAILABLE:
        info['cupy_version'] = cp.__version__
        try:
            info['cuda_version'] = cp.cuda.runtime.runtimeGetVersion()
            info['gpu_count'] = cp.cuda.runtime.getDeviceCount()
            if info['gpu_count'] > 0:
                props = cp.cuda.runtime.getDeviceProperties(0)
                info['gpu_name'] = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
                info['gpu_memory'] = props['totalGlobalMem']
        except Exception:
            pass
    
    return info


def print_device_info():
    """Print device information."""
    info = get_device_info()
    print("=" * 50)
    print("NexNet Device Information")
    print("=" * 50)
    print(f"Current Backend: {info['backend']}")
    print(f"NumPy Version: {info['numpy_version']}")
    print(f"CuPy Available: {info['cupy_available']}")
    
    if info['cupy_available']:
        print(f"CuPy Version: {info.get('cupy_version', 'N/A')}")
        print(f"CUDA Version: {info.get('cuda_version', 'N/A')}")
        print(f"GPU Count: {info.get('gpu_count', 'N/A')}")
        if 'gpu_name' in info:
            print(f"GPU Name: {info['gpu_name']}")
            print(f"GPU Memory: {info['gpu_memory'] / (1024**3):.2f} GB")
    print("=" * 50)


__all__ = [
    'is_cupy_available',
    'get_backend',
    'set_backend',
    'get_array_module',
    'to_cpu',
    'to_gpu',
    'to_device',
    'zeros',
    'ones',
    'zeros_like',
    'ones_like',
    'array',
    'asarray',
    'random_randn',
    'random_rand',
    'random_uniform',
    'random_normal',
    'random_randint',
    'random_permutation',
    'random_choice',
    'seed',
    'arange',
    'linspace',
    'eye',
    'concatenate',
    'stack',
    'vstack',
    'hstack',
    'split',
    'clip',
    'where',
    'maximum',
    'minimum',
    'abs',
    'sign',
    'sqrt',
    'square',
    'floor',
    'ceil',
    'round',
    'power',
    'exp',
    'log',
    'log2',
    'log10',
    'sin',
    'cos',
    'tan',
    'tanh',
    'sinh',
    'cosh',
    'sum',
    'mean',
    'std',
    'var',
    'max',
    'min',
    'argmax',
    'argmin',
    'argsort',
    'sort',
    'prod',
    'cumsum',
    'dot',
    'matmul',
    'tensordot',
    'einsum',
    'transpose',
    'reshape',
    'expand_dims',
    'squeeze',
    'swapaxes',
    'moveaxis',
    'broadcast_to',
    'tile',
    'repeat',
    'flip',
    'roll',
    'pad',
    'allclose',
    'isnan',
    'isinf',
    'isfinite',
    'nan_to_num',
    'take_along_axis',
    'put_along_axis',
    'unique',
    'bincount',
    'norm',
    'triu',
    'tril',
    'diag',
    'trace',
    'diagflat',
    'outer',
    'erf',
    'DeviceContext',
    'use_gpu',
    'use_cpu',
    'get_device_info',
    'print_device_info',
]
