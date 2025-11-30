"""
NexNet Core Package.

Contains base classes, module definitions, and backend abstraction
for GPU/CPU computation.
"""

from .base import Layer, Activation, Loss, Optimizer
from .module import Module, Parameter, init_weights
from .backend import (
    get_array_module,
    get_backend,
    set_backend,
    is_cupy_available,
    to_cpu,
    to_gpu,
    to_device,
    use_gpu,
    use_cpu,
    get_device_info,
    print_device_info,
)

__all__ = [
    'Layer',
    'Activation', 
    'Loss',
    'Optimizer',
    'Module',
    'Parameter',
    'init_weights',
    'get_array_module',
    'get_backend',
    'set_backend',
    'is_cupy_available',
    'to_cpu',
    'to_gpu',
    'to_device',
    'use_gpu',
    'use_cpu',
    'get_device_info',
    'print_device_info',
]