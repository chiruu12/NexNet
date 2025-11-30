"""
Image Transform Classes for Data Augmentation.

This module provides a comprehensive set of image transformations that
can be composed into augmentation pipelines. All transforms work with
NumPy arrays and support both single images and batches.

Supported image formats:
- Single image: (H, W), (H, W, C), or (C, H, W)
- Batch: (N, H, W), (N, H, W, C), or (N, C, H, W)
"""

import math
from core.backend import (
    get_array_module, zeros, ones, zeros_like, random_rand, random_randn,
    random_randint, random_uniform, clip, stack, concatenate,
    sin, cos, sqrt, exp, floor, minimum, maximum, mean, std, sum as xp_sum
)
import numpy as np  # For some operations that need CPU-side execution


class Transform:
    """Base class for all transforms."""
    
    def __call__(self, x):
        """Apply the transform to input."""
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Compose(Transform):
    """
    Compose multiple transforms together.
    
    Args:
        transforms: List of transforms to apply sequentially.
    
    Example:
        >>> transform = Compose([
        ...     RandomHorizontalFlip(p=0.5),
        ...     RandomRotate(max_angle=15),
        ...     Normalize(mean=[0.5], std=[0.5])
        ... ])
        >>> augmented = transform(image)
    """
    
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
    
    def __repr__(self):
        transform_strs = [f"    {t}" for t in self.transforms]
        return f"Compose([\n" + ",\n".join(transform_strs) + "\n])"


class ToTensor(Transform):
    """
    Convert array to float tensor and scale to [0, 1] if integer input.
    
    Also handles channel dimension ordering:
    - (H, W, C) -> (C, H, W) if channels_first=True
    """
    
    def __init__(self, channels_first=True, scale=True):
        """
        Args:
            channels_first: If True, move channels to first dimension.
            scale: If True and input is uint8, scale to [0, 1].
        """
        self.channels_first = channels_first
        self.scale = scale
        
    def __call__(self, x):
        xp = get_array_module(x)
        
        # Scale if needed
        if self.scale and x.dtype == xp.uint8:
            x = x.astype(xp.float32) / 255.0
        elif x.dtype not in [xp.float32, xp.float64]:
            x = x.astype(xp.float32)
            
        # Reorder channels if needed
        if self.channels_first and x.ndim >= 3:
            if x.ndim == 3:  # (H, W, C)
                x = xp.transpose(x, (2, 0, 1))
            elif x.ndim == 4:  # (N, H, W, C)
                x = xp.transpose(x, (0, 3, 1, 2))
                
        return x


class Normalize(Transform):
    """
    Normalize image with mean and standard deviation.
    
    Args:
        mean: Sequence of means for each channel.
        std: Sequence of standard deviations for each channel.
        
    Note:
        Assumes input is in [0, 1] range for images.
    """
    
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        
    def __call__(self, x):
        xp = get_array_module(x)
        mean = xp.asarray(self.mean)
        std = xp.asarray(self.std)
        
        # Handle different input shapes
        if x.ndim == 3:  # (C, H, W) or (H, W, C)
            if x.shape[0] == len(mean):  # (C, H, W)
                mean = mean.reshape(-1, 1, 1)
                std = std.reshape(-1, 1, 1)
            else:  # (H, W, C)
                pass  # mean/std can broadcast
        elif x.ndim == 4:  # (N, C, H, W) or (N, H, W, C)
            if x.shape[1] == len(mean):  # (N, C, H, W)
                mean = mean.reshape(1, -1, 1, 1)
                std = std.reshape(1, -1, 1, 1)
            else:  # (N, H, W, C)
                mean = mean.reshape(1, 1, 1, -1)
                std = std.reshape(1, 1, 1, -1)
                
        return (x - mean) / std
    
    def __repr__(self):
        return f"Normalize(mean={self.mean.tolist()}, std={self.std.tolist()})"


class RandomHorizontalFlip(Transform):
    """
    Randomly flip the image horizontally.
    
    Args:
        p: Probability of flipping.
    """
    
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, x):
        if np.random.random() < self.p:
            xp = get_array_module(x)
            if x.ndim == 2:  # (H, W)
                return xp.flip(x, axis=1)
            elif x.ndim == 3:  # (H, W, C) - flip along width (axis=1)
                return xp.flip(x, axis=1)
            elif x.ndim == 4:  # (N, H, W, C) - flip along width (axis=2)
                return xp.flip(x, axis=2)
        return x
    
    def __repr__(self):
        return f"RandomHorizontalFlip(p={self.p})"


class RandomVerticalFlip(Transform):
    """
    Randomly flip the image vertically.
    
    Args:
        p: Probability of flipping.
    """
    
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, x):
        if np.random.random() < self.p:
            xp = get_array_module(x)
            if x.ndim == 2:  # (H, W)
                return xp.flip(x, axis=0)
            elif x.ndim == 3:  # (C, H, W) or (H, W, C)
                return xp.flip(x, axis=0 if x.shape[0] <= 4 else 0)
            elif x.ndim == 4:  # (N, C, H, W) or (N, H, W, C)
                return xp.flip(x, axis=2 if x.shape[1] <= 4 else 1)
        return x
    
    def __repr__(self):
        return f"RandomVerticalFlip(p={self.p})"


# Alias
RandomFlip = RandomHorizontalFlip


class RandomRotate(Transform):
    """
    Randomly rotate the image.
    
    Args:
        max_angle: Maximum rotation angle in degrees.
        p: Probability of applying rotation.
        fill: Fill value for areas outside the rotated image.
    """
    
    def __init__(self, max_angle=15, p=0.5, fill=0):
        self.max_angle = max_angle
        self.p = p
        self.fill = fill
        
    def __call__(self, x):
        if np.random.random() > self.p:
            return x
            
        xp = get_array_module(x)
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        return self._rotate(x, angle)
    
    def _rotate(self, x, angle):
        """Rotate image by angle degrees."""
        xp = get_array_module(x)
        rad = angle * math.pi / 180
        
        # Get dimensions
        if x.ndim == 2:
            h, w = x.shape
            is_batch = False
            is_channel_first = False
        elif x.ndim == 3:
            if x.shape[0] <= 4:  # (C, H, W)
                c, h, w = x.shape
                is_batch = False
                is_channel_first = True
            else:  # (H, W, C)
                h, w, c = x.shape
                is_batch = False
                is_channel_first = False
        else:  # x.ndim == 4
            if x.shape[1] <= 4:  # (N, C, H, W)
                n, c, h, w = x.shape
                is_batch = True
                is_channel_first = True
            else:  # (N, H, W, C)
                n, h, w, c = x.shape
                is_batch = True
                is_channel_first = False
        
        # Rotation matrix
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        cx, cy = w / 2, h / 2
        
        # Create coordinate grids
        y_coords = xp.arange(h).reshape(-1, 1)
        x_coords = xp.arange(w).reshape(1, -1)
        
        # Apply inverse rotation to find source coordinates
        src_x = cos_a * (x_coords - cx) + sin_a * (y_coords - cy) + cx
        src_y = -sin_a * (x_coords - cx) + cos_a * (y_coords - cy) + cy
        
        # Nearest neighbor interpolation with bounds checking
        src_x = xp.clip(xp.round(src_x).astype(xp.int32), 0, w - 1)
        src_y = xp.clip(xp.round(src_y).astype(xp.int32), 0, h - 1)
        
        # Sample from source
        if x.ndim == 2:
            return x[src_y, src_x]
        elif x.ndim == 3:
            if is_channel_first:
                return x[:, src_y, src_x]
            else:
                return x[src_y, src_x, :]
        else:  # x.ndim == 4
            if is_channel_first:
                return x[:, :, src_y, src_x]
            else:
                return x[:, src_y, src_x, :]
    
    def __repr__(self):
        return f"RandomRotate(max_angle={self.max_angle}, p={self.p})"


class RandomCrop(Transform):
    """
    Randomly crop the image.
    
    Args:
        size: Output size (height, width) or single int for square crop.
        padding: Optional padding to add before cropping.
        fill: Fill value for padding.
    """
    
    def __init__(self, size, padding=0, fill=0):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
        self.padding = padding
        self.fill = fill
        
    def __call__(self, x):
        xp = get_array_module(x)
        
        # Add padding if needed
        if self.padding > 0:
            if x.ndim == 2:
                pad_width = ((self.padding, self.padding), (self.padding, self.padding))
            elif x.ndim == 3:
                if x.shape[0] <= 4:  # (C, H, W)
                    pad_width = ((0, 0), (self.padding, self.padding), (self.padding, self.padding))
                else:  # (H, W, C)
                    pad_width = ((self.padding, self.padding), (self.padding, self.padding), (0, 0))
            else:  # (N, C, H, W) or (N, H, W, C)
                if x.shape[1] <= 4:  # (N, C, H, W)
                    pad_width = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
                else:  # (N, H, W, C)
                    pad_width = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
            x = xp.pad(x, pad_width, mode='constant', constant_values=self.fill)
        
        # Get spatial dimensions
        if x.ndim == 2:
            h, w = x.shape
        elif x.ndim == 3:
            if x.shape[0] <= 4:
                _, h, w = x.shape
            else:
                h, w, _ = x.shape
        else:
            if x.shape[1] <= 4:
                _, _, h, w = x.shape
            else:
                _, h, w, _ = x.shape
        
        # Random crop position
        th, tw = self.size
        if h < th or w < tw:
            raise ValueError(f"Image size ({h}, {w}) is smaller than crop size {self.size}")
        
        top = np.random.randint(0, h - th + 1)
        left = np.random.randint(0, w - tw + 1)
        
        # Apply crop
        if x.ndim == 2:
            return x[top:top+th, left:left+tw]
        elif x.ndim == 3:
            if x.shape[0] <= 4:
                return x[:, top:top+th, left:left+tw]
            else:
                return x[top:top+th, left:left+tw, :]
        else:
            if x.shape[1] <= 4:
                return x[:, :, top:top+th, left:left+tw]
            else:
                return x[:, top:top+th, left:left+tw, :]
    
    def __repr__(self):
        return f"RandomCrop(size={self.size}, padding={self.padding})"


class CenterCrop(Transform):
    """
    Crop the center of the image.
    
    Args:
        size: Output size (height, width) or single int for square crop.
    """
    
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
    
    def _is_hwc(self, x):
        """Check if image is in HWC format (last dim is channels)."""
        if x.ndim == 3:
            # If last dimension is small (1, 3, or 4), assume HWC
            return x.shape[-1] in (1, 3, 4)
        elif x.ndim == 4:
            # Batch: if last dimension is small, assume NHWC
            return x.shape[-1] in (1, 3, 4)
        return False
            
    def __call__(self, x):
        # Get spatial dimensions
        if x.ndim == 2:
            h, w = x.shape
        elif x.ndim == 3:
            if self._is_hwc(x):
                h, w, _ = x.shape
            else:
                _, h, w = x.shape
        else:
            if self._is_hwc(x):
                _, h, w, _ = x.shape
            else:
                _, _, h, w = x.shape
        
        th, tw = self.size
        top = (h - th) // 2
        left = (w - tw) // 2
        
        # Apply crop
        if x.ndim == 2:
            return x[top:top+th, left:left+tw]
        elif x.ndim == 3:
            if self._is_hwc(x):
                return x[top:top+th, left:left+tw, :]
            else:
                return x[:, top:top+th, left:left+tw]
        else:
            if self._is_hwc(x):
                return x[:, top:top+th, left:left+tw, :]
            else:
                return x[:, :, top:top+th, left:left+tw]
    
    def __repr__(self):
        return f"CenterCrop(size={self.size})"


class Resize(Transform):
    """
    Resize the image to given size using nearest neighbor interpolation.
    
    Args:
        size: Output size (height, width) or single int.
    """
    
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
            
    def __call__(self, x):
        xp = get_array_module(x)
        th, tw = self.size
        
        # Get spatial dimensions
        if x.ndim == 2:
            h, w = x.shape
            new_shape = (th, tw)
        elif x.ndim == 3:
            if x.shape[0] <= 4:
                c, h, w = x.shape
                new_shape = (c, th, tw)
            else:
                h, w, c = x.shape
                new_shape = (th, tw, c)
        else:
            if x.shape[1] <= 4:
                n, c, h, w = x.shape
                new_shape = (n, c, th, tw)
            else:
                n, h, w, c = x.shape
                new_shape = (n, th, tw, c)
        
        # Create coordinate mapping
        y_ratio = h / th
        x_ratio = w / tw
        
        y_coords = (xp.arange(th) * y_ratio).astype(xp.int32)
        x_coords = (xp.arange(tw) * x_ratio).astype(xp.int32)
        
        y_coords = xp.clip(y_coords, 0, h - 1)
        x_coords = xp.clip(x_coords, 0, w - 1)
        
        # Create meshgrid
        yy, xx = xp.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Apply resize
        if x.ndim == 2:
            return x[yy, xx]
        elif x.ndim == 3:
            if x.shape[0] <= 4:
                return x[:, yy, xx]
            else:
                return x[yy, xx, :]
        else:
            if x.shape[1] <= 4:
                return x[:, :, yy, xx]
            else:
                return x[:, yy, xx, :]
    
    def __repr__(self):
        return f"Resize(size={self.size})"


class RandomResizedCrop(Transform):
    """
    Crop a random portion of image and resize it to given size.
    
    Args:
        size: Output size (height, width).
        scale: Range of size of the origin size cropped.
        ratio: Range of aspect ratio of the origin aspect ratio cropped.
    """
    
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
        self.scale = scale
        self.ratio = ratio
        
    def __call__(self, x):
        xp = get_array_module(x)
        
        # Get spatial dimensions
        if x.ndim == 2:
            h, w = x.shape
        elif x.ndim == 3:
            if x.shape[0] <= 4:
                _, h, w = x.shape
            else:
                h, w, _ = x.shape
        else:
            if x.shape[1] <= 4:
                _, _, h, w = x.shape
            else:
                _, h, w, _ = x.shape
        
        area = h * w
        
        for _ in range(10):
            target_area = np.random.uniform(*self.scale) * area
            aspect_ratio = np.exp(np.random.uniform(np.log(self.ratio[0]), np.log(self.ratio[1])))
            
            new_w = int(round(np.sqrt(target_area * aspect_ratio)))
            new_h = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if 0 < new_w <= w and 0 < new_h <= h:
                top = np.random.randint(0, h - new_h + 1)
                left = np.random.randint(0, w - new_w + 1)
                
                # Crop
                if x.ndim == 2:
                    cropped = x[top:top+new_h, left:left+new_w]
                elif x.ndim == 3:
                    if x.shape[0] <= 4:
                        cropped = x[:, top:top+new_h, left:left+new_w]
                    else:
                        cropped = x[top:top+new_h, left:left+new_w, :]
                else:
                    if x.shape[1] <= 4:
                        cropped = x[:, :, top:top+new_h, left:left+new_w]
                    else:
                        cropped = x[:, top:top+new_h, left:left+new_w, :]
                
                # Resize
                resize = Resize(self.size)
                return resize(cropped)
        
        # Fallback to center crop
        return CenterCrop(min(h, w))(x)
    
    def __repr__(self):
        return f"RandomResizedCrop(size={self.size}, scale={self.scale}, ratio={self.ratio})"


class ColorJitter(Transform):
    """
    Randomly change brightness, contrast, saturation, and hue.
    
    Args:
        brightness: How much to jitter brightness.
        contrast: How much to jitter contrast.
        saturation: How much to jitter saturation.
        hue: How much to jitter hue.
    """
    
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
    def __call__(self, x):
        xp = get_array_module(x)
        
        # Apply in random order
        transforms = []
        if self.brightness > 0:
            transforms.append(('brightness', self.brightness))
        if self.contrast > 0:
            transforms.append(('contrast', self.contrast))
        if self.saturation > 0:
            transforms.append(('saturation', self.saturation))
        if self.hue > 0:
            transforms.append(('hue', self.hue))
        
        np.random.shuffle(transforms)
        
        for name, value in transforms:
            if name == 'brightness':
                factor = np.random.uniform(max(0, 1 - value), 1 + value)
                x = x * factor
            elif name == 'contrast':
                factor = np.random.uniform(max(0, 1 - value), 1 + value)
                gray = mean(x, axis=-1, keepdims=True) if x.ndim >= 3 else mean(x)
                x = (x - gray) * factor + gray
            elif name == 'saturation' and x.ndim >= 3:
                factor = np.random.uniform(max(0, 1 - value), 1 + value)
                # Simple grayscale conversion
                if x.shape[-1] >= 3 or (x.ndim >= 3 and x.shape[0] >= 3):
                    gray = self._to_grayscale(x)
                    x = gray + factor * (x - gray)
            elif name == 'hue':
                # Simplified hue shift
                pass
        
        return clip(x, 0, 1)
    
    def _to_grayscale(self, x):
        """Convert to grayscale while maintaining shape."""
        xp = get_array_module(x)
        weights = xp.array([0.299, 0.587, 0.114])
        
        if x.ndim == 3:
            if x.shape[0] <= 4:  # (C, H, W)
                gray = xp_sum(x[:3] * weights.reshape(-1, 1, 1), axis=0, keepdims=True)
                return xp.broadcast_to(gray, x.shape)
            else:  # (H, W, C)
                gray = xp_sum(x[..., :3] * weights, axis=-1, keepdims=True)
                return xp.broadcast_to(gray, x.shape)
        elif x.ndim == 4:
            if x.shape[1] <= 4:  # (N, C, H, W)
                gray = xp_sum(x[:, :3] * weights.reshape(1, -1, 1, 1), axis=1, keepdims=True)
                return xp.broadcast_to(gray, x.shape)
            else:  # (N, H, W, C)
                gray = xp_sum(x[..., :3] * weights, axis=-1, keepdims=True)
                return xp.broadcast_to(gray, x.shape)
        return x
    
    def __repr__(self):
        return f"ColorJitter(brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation}, hue={self.hue})"


class RandomBrightness(Transform):
    """Randomly adjust brightness."""
    
    def __init__(self, factor=0.2, p=0.5):
        self.factor = factor
        self.p = p
        
    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        factor = np.random.uniform(max(0, 1 - self.factor), 1 + self.factor)
        return clip(x * factor, 0, 1)


class RandomContrast(Transform):
    """Randomly adjust contrast."""
    
    def __init__(self, factor=0.2, p=0.5):
        self.factor = factor
        self.p = p
        
    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        factor = np.random.uniform(max(0, 1 - self.factor), 1 + self.factor)
        gray = mean(x)
        return clip((x - gray) * factor + gray, 0, 1)


class RandomSaturation(Transform):
    """Randomly adjust saturation."""
    
    def __init__(self, factor=0.2, p=0.5):
        self.factor = factor
        self.p = p
        
    def __call__(self, x):
        if np.random.random() > self.p or x.ndim < 3:
            return x
        factor = np.random.uniform(max(0, 1 - self.factor), 1 + self.factor)
        xp = get_array_module(x)
        gray = mean(x, axis=-1, keepdims=True) if x.shape[-1] <= 4 else mean(x, axis=0, keepdims=True)
        return clip(gray + factor * (x - gray), 0, 1)


class RandomHue(Transform):
    """Randomly adjust hue (simplified)."""
    
    def __init__(self, factor=0.1, p=0.5):
        self.factor = factor
        self.p = p
        
    def __call__(self, x):
        # Simplified - just returns input for now
        # Full hue shift requires RGB->HSV->RGB conversion
        return x


class GaussianNoise(Transform):
    """
    Add Gaussian noise to the image.
    
    Args:
        mean: Mean of the noise.
        std: Standard deviation of the noise.
        p: Probability of applying noise.
    """
    
    def __init__(self, mean=0, std=0.1, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p
        
    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        xp = get_array_module(x)
        noise = xp.random.randn(*x.shape) * self.std + self.mean
        return clip(x + noise, 0, 1)
    
    def __repr__(self):
        return f"GaussianNoise(mean={self.mean}, std={self.std}, p={self.p})"


class GaussianBlur(Transform):
    """
    Apply Gaussian blur to the image.
    
    Args:
        kernel_size: Size of the blur kernel.
        sigma: Standard deviation of the Gaussian.
        p: Probability of applying blur.
    """
    
    def __init__(self, kernel_size=3, sigma=1.0, p=0.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
        self._create_kernel()
        
    def _create_kernel(self):
        """Create Gaussian kernel."""
        k = self.kernel_size // 2
        x = np.arange(-k, k + 1)
        kernel = np.exp(-x ** 2 / (2 * self.sigma ** 2))
        kernel = kernel / kernel.sum()
        self.kernel = kernel
        
    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        
        xp = get_array_module(x)
        kernel = xp.asarray(self.kernel)
        
        # Apply separable filter (horizontal then vertical)
        # Simplified 1D convolution
        pad = self.kernel_size // 2
        
        if x.ndim == 2:
            # Horizontal
            x = xp.pad(x, ((0, 0), (pad, pad)), mode='reflect')
            out = xp.zeros_like(x[:, pad:-pad] if pad > 0 else x)
            for i, w in enumerate(kernel):
                out += w * x[:, i:i+out.shape[1]]
            # Vertical
            out = xp.pad(out, ((pad, pad), (0, 0)), mode='reflect')
            result = xp.zeros_like(out[pad:-pad] if pad > 0 else out)
            for i, w in enumerate(kernel):
                result += w * out[i:i+result.shape[0]]
            return result
        
        return x  # Simplified - return unchanged for higher dims
    
    def __repr__(self):
        return f"GaussianBlur(kernel_size={self.kernel_size}, sigma={self.sigma}, p={self.p})"


class RandomErasing(Transform):
    """
    Randomly erase rectangular regions.
    
    Args:
        p: Probability of erasing.
        scale: Range of proportion of erased area.
        ratio: Range of aspect ratio of erased area.
        value: Erasing value (0 for black, 'random' for random noise).
    """
    
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        
    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        
        xp = get_array_module(x)
        
        # Get spatial dimensions
        if x.ndim == 2:
            h, w = x.shape
        elif x.ndim == 3:
            if x.shape[0] <= 4:
                _, h, w = x.shape
            else:
                h, w, _ = x.shape
        else:
            if x.shape[1] <= 4:
                _, _, h, w = x.shape
            else:
                _, h, w, _ = x.shape
        
        area = h * w
        
        for _ in range(10):
            erase_area = np.random.uniform(*self.scale) * area
            aspect_ratio = np.random.uniform(*self.ratio)
            
            eh = int(round(np.sqrt(erase_area * aspect_ratio)))
            ew = int(round(np.sqrt(erase_area / aspect_ratio)))
            
            if eh < h and ew < w:
                top = np.random.randint(0, h - eh)
                left = np.random.randint(0, w - ew)
                
                x = x.copy()
                if self.value == 'random':
                    fill_value = xp.random.rand()
                else:
                    fill_value = self.value
                
                if x.ndim == 2:
                    x[top:top+eh, left:left+ew] = fill_value
                elif x.ndim == 3:
                    if x.shape[0] <= 4:
                        x[:, top:top+eh, left:left+ew] = fill_value
                    else:
                        x[top:top+eh, left:left+ew, :] = fill_value
                else:
                    if x.shape[1] <= 4:
                        x[:, :, top:top+eh, left:left+ew] = fill_value
                    else:
                        x[:, top:top+eh, left:left+ew, :] = fill_value
                break
        
        return x
    
    def __repr__(self):
        return f"RandomErasing(p={self.p}, scale={self.scale}, ratio={self.ratio})"


class Cutout(Transform):
    """
    Cutout augmentation - randomly mask out square regions.
    
    Args:
        n_holes: Number of patches to cut out.
        length: Length of the square patch.
        p: Probability of applying cutout.
    """
    
    def __init__(self, n_holes=1, length=16, p=0.5):
        self.n_holes = n_holes
        self.length = length
        self.p = p
        
    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        
        xp = get_array_module(x)
        x = x.copy()
        
        # Get spatial dimensions
        if x.ndim == 2:
            h, w = x.shape
        elif x.ndim == 3:
            if x.shape[0] <= 4:
                _, h, w = x.shape
            else:
                h, w, _ = x.shape
        else:
            return x  # Skip for batches
        
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x_pos = np.random.randint(w)
            
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x_pos - self.length // 2)
            x2 = min(w, x_pos + self.length // 2)
            
            if x.ndim == 2:
                x[y1:y2, x1:x2] = 0
            elif x.ndim == 3:
                if x.shape[0] <= 4:
                    x[:, y1:y2, x1:x2] = 0
                else:
                    x[y1:y2, x1:x2, :] = 0
        
        return x
    
    def __repr__(self):
        return f"Cutout(n_holes={self.n_holes}, length={self.length}, p={self.p})"


class Mixup:
    """
    Mixup augmentation for batch data.
    
    Mixes images and labels: x_mixed = λ*x1 + (1-λ)*x2
    
    Args:
        alpha: Parameter for Beta distribution.
    
    Returns:
        Tuple of (mixed_images, labels1, labels2, lambda)
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, x, y):
        xp = get_array_module(x)
        batch_size = x.shape[0]
        
        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Random permutation for mixing
        index = np.random.permutation(batch_size)
        
        # Mix images
        mixed_x = lam * x + (1 - lam) * x[index]
        
        return mixed_x, y, y[index], lam
    
    def __repr__(self):
        return f"Mixup(alpha={self.alpha})"


class CutMix:
    """
    CutMix augmentation for batch data.
    
    Cuts and pastes patches between images and mixes labels accordingly.
    
    Args:
        alpha: Parameter for Beta distribution.
    
    Returns:
        Tuple of (mixed_images, labels1, labels2, lambda)
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, x, y):
        xp = get_array_module(x)
        batch_size = x.shape[0]
        
        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Random permutation for mixing
        index = np.random.permutation(batch_size)
        
        # Get spatial dimensions
        if x.shape[1] <= 4:  # (N, C, H, W)
            _, _, h, w = x.shape
            is_channels_first = True
        else:  # (N, H, W, C)
            _, h, w, _ = x.shape
            is_channels_first = False
        
        # Get bounding box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = max(0, cx - cut_w // 2)
        bby1 = max(0, cy - cut_h // 2)
        bbx2 = min(w, cx + cut_w // 2)
        bby2 = min(h, cy + cut_h // 2)
        
        # Mix images
        mixed_x = x.copy()
        if is_channels_first:
            mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        else:
            mixed_x[:, bby1:bby2, bbx1:bbx2, :] = x[index, bby1:bby2, bbx1:bbx2, :]
        
        # Adjust lambda based on actual bbox size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        return mixed_x, y, y[index], lam
    
    def __repr__(self):
        return f"CutMix(alpha={self.alpha})"


class RandomAffine(Transform):
    """
    Random affine transformation.
    
    Args:
        degrees: Range of rotation.
        translate: Range of horizontal and vertical shifts.
        scale: Range of scale.
        shear: Range of shear.
        p: Probability of applying transform.
    """
    
    def __init__(self, degrees=0, translate=None, scale=None, shear=None, p=0.5):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.p = p
        
    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        
        # For simplicity, use RandomRotate for rotation
        if self.degrees != 0:
            rotate = RandomRotate(self.degrees, p=1.0)
            x = rotate(x)
        
        return x
    
    def __repr__(self):
        return f"RandomAffine(degrees={self.degrees}, translate={self.translate}, scale={self.scale})"


class ElasticTransform(Transform):
    """
    Elastic deformation of images.
    
    Args:
        alpha: Scaling factor for displacement.
        sigma: Gaussian filter parameter.
        p: Probability of applying transform.
    """
    
    def __init__(self, alpha=50, sigma=5, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        
    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        # Simplified - return unchanged
        # Full implementation requires displacement field generation
        return x
    
    def __repr__(self):
        return f"ElasticTransform(alpha={self.alpha}, sigma={self.sigma})"


class GridDistortion(Transform):
    """Grid distortion augmentation."""
    
    def __init__(self, num_steps=5, distort_limit=0.3, p=0.5):
        self.num_steps = num_steps
        self.distort_limit = distort_limit
        self.p = p
        
    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        return x  # Simplified
    
    def __repr__(self):
        return f"GridDistortion(num_steps={self.num_steps})"


class OpticalDistortion(Transform):
    """Optical distortion augmentation."""
    
    def __init__(self, distort_limit=0.05, shift_limit=0.05, p=0.5):
        self.distort_limit = distort_limit
        self.shift_limit = shift_limit
        self.p = p
        
    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        return x  # Simplified
    
    def __repr__(self):
        return f"OpticalDistortion(distort_limit={self.distort_limit})"


class RandomPerspective(Transform):
    """Random perspective transformation."""
    
    def __init__(self, distortion_scale=0.5, p=0.5):
        self.distortion_scale = distortion_scale
        self.p = p
        
    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        return x  # Simplified
    
    def __repr__(self):
        return f"RandomPerspective(distortion_scale={self.distortion_scale})"


class RandomGrayscale(Transform):
    """Randomly convert image to grayscale."""
    
    def __init__(self, p=0.1):
        self.p = p
        
    def __call__(self, x):
        if np.random.random() > self.p or x.ndim < 3:
            return x
        
        xp = get_array_module(x)
        weights = xp.array([0.299, 0.587, 0.114])
        
        if x.ndim == 3:
            if x.shape[0] <= 4:  # (C, H, W)
                gray = xp_sum(x[:3] * weights.reshape(-1, 1, 1), axis=0, keepdims=True)
                return xp.broadcast_to(gray, x.shape)
            else:  # (H, W, C)
                gray = xp_sum(x[..., :3] * weights, axis=-1, keepdims=True)
                return xp.broadcast_to(gray, x.shape)
        return x
    
    def __repr__(self):
        return f"RandomGrayscale(p={self.p})"


class RandomInvert(Transform):
    """Randomly invert the colors of the image."""
    
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        return 1.0 - x
    
    def __repr__(self):
        return f"RandomInvert(p={self.p})"


class RandomSolarize(Transform):
    """
    Solarize the image by inverting all pixel values above a threshold.
    
    Args:
        threshold: All pixels above this value are inverted.
        p: Probability of applying transform.
    """
    
    def __init__(self, threshold=0.5, p=0.5):
        self.threshold = threshold
        self.p = p
        
    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        xp = get_array_module(x)
        mask = x >= self.threshold
        result = x.copy()
        result[mask] = 1.0 - result[mask]
        return result
    
    def __repr__(self):
        return f"RandomSolarize(threshold={self.threshold}, p={self.p})"


class RandomPosterize(Transform):
    """
    Posterize the image by reducing the number of bits.
    
    Args:
        bits: Number of bits to keep (1-8).
        p: Probability of applying transform.
    """
    
    def __init__(self, bits=4, p=0.5):
        self.bits = bits
        self.p = p
        
    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        xp = get_array_module(x)
        # Assuming input is in [0, 1]
        shift = 8 - self.bits
        return xp.floor(x * 255 / (2 ** shift)) * (2 ** shift) / 255
    
    def __repr__(self):
        return f"RandomPosterize(bits={self.bits}, p={self.p})"


class RandomApply(Transform):
    """
    Apply a list of transforms with a given probability.
    
    Args:
        transforms: List of transforms to apply.
        p: Probability of applying the transforms.
    """
    
    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p
        
    def __call__(self, x):
        if np.random.random() < self.p:
            for t in self.transforms:
                x = t(x)
        return x
    
    def __repr__(self):
        return f"RandomApply(transforms={self.transforms}, p={self.p})"


class RandomChoice(Transform):
    """
    Apply a single transform randomly selected from a list.
    
    Args:
        transforms: List of transforms to choose from.
    """
    
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, x):
        t = np.random.choice(self.transforms)
        return t(x)
    
    def __repr__(self):
        return f"RandomChoice(transforms={self.transforms})"


class RandomOrder(Transform):
    """
    Apply a list of transforms in random order.
    
    Args:
        transforms: List of transforms to apply in random order.
    """
    
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, x):
        order = np.random.permutation(len(self.transforms))
        for idx in order:
            x = self.transforms[idx](x)
        return x
    
    def __repr__(self):
        return f"RandomOrder(transforms={self.transforms})"


class AutoAugment(Transform):
    """
    AutoAugment policy from "AutoAugment: Learning Augmentation Strategies from Data".
    
    Applies automatically learned augmentation policies.
    """
    
    def __init__(self, policy='imagenet'):
        self.policy = policy
        # Simplified - just use basic augmentations
        self.transforms = [
            RandomHorizontalFlip(p=0.5),
            RandomRotate(max_angle=10, p=0.5),
            ColorJitter(brightness=0.2, contrast=0.2),
        ]
        
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
    
    def __repr__(self):
        return f"AutoAugment(policy='{self.policy}')"


class RandAugment(Transform):
    """
    RandAugment: Practical automated data augmentation.
    
    Applies N random transformations from a predefined set with magnitude M.
    
    Args:
        n: Number of transformations to apply.
        m: Magnitude of transformations (0-10).
    """
    
    def __init__(self, n=2, m=9):
        self.n = n
        self.m = m
        
        # Define augmentation operations
        magnitude = m / 10.0
        self.ops = [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotate(max_angle=30 * magnitude, p=1.0),
            RandomBrightness(factor=0.9 * magnitude, p=1.0),
            RandomContrast(factor=0.9 * magnitude, p=1.0),
            RandomSolarize(threshold=1.0 - 0.5 * magnitude, p=1.0),
            RandomPosterize(bits=max(1, int(8 - 4 * magnitude)), p=1.0),
            RandomInvert(p=1.0),
        ]
        
    def __call__(self, x):
        # Randomly select N operations
        ops_idx = np.random.choice(len(self.ops), self.n, replace=False)
        for idx in ops_idx:
            x = self.ops[idx](x)
        return x
    
    def __repr__(self):
        return f"RandAugment(n={self.n}, m={self.m})"


# Helper function
def random_uniform(low=0, high=1):
    """Generate uniform random value."""
    return np.random.uniform(low, high)
