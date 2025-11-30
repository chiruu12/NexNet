"""
Image Augmentor - High-level interface for data augmentation.

Provides easy-to-use augmentation pipelines for training deep learning models.
"""

import numpy as np
from core.backend import get_array_module

from .transforms import (
    Compose, ToTensor, Normalize,
    RandomHorizontalFlip, RandomVerticalFlip, RandomRotate,
    RandomCrop, CenterCrop, Resize, RandomResizedCrop,
    ColorJitter, GaussianNoise, GaussianBlur,
    RandomErasing, Cutout, Mixup, CutMix,
    RandAugment, AutoAugment
)


class ImageAugmentor:
    """
    High-level image augmentation interface.
    
    Provides preset augmentation strategies for common use cases and
    allows custom augmentation pipelines.
    
    Example:
        >>> # Use preset for CIFAR-10
        >>> augmentor = ImageAugmentor.for_cifar10()
        >>> augmented = augmentor(images)
        
        >>> # Use preset for ImageNet
        >>> augmentor = ImageAugmentor.for_imagenet(train=True)
        >>> augmented = augmentor(images)
        
        >>> # Custom augmentation
        >>> augmentor = ImageAugmentor(
        ...     horizontal_flip=True,
        ...     rotation=15,
        ...     color_jitter=True,
        ...     normalize=True
        ... )
        >>> augmented = augmentor(images)
    """
    
    def __init__(
        self,
        # Geometric transforms
        horizontal_flip=False,
        vertical_flip=False,
        rotation=0,
        crop_size=None,
        crop_padding=0,
        resize=None,
        random_resized_crop=None,
        # Color transforms
        color_jitter=False,
        brightness=0,
        contrast=0,
        saturation=0,
        hue=0,
        grayscale_prob=0,
        # Noise and blur
        gaussian_noise=0,
        gaussian_blur=False,
        blur_kernel_size=3,
        # Regularization
        random_erasing=False,
        cutout=False,
        cutout_length=16,
        # Normalization
        normalize=False,
        mean=None,
        std=None,
        # Advanced
        randaugment=False,
        randaugment_n=2,
        randaugment_m=9,
        autoaugment=False,
        # Output
        to_tensor=False,
        channels_first=True,
    ):
        """
        Initialize ImageAugmentor with specified augmentation options.
        
        Args:
            horizontal_flip: Apply random horizontal flip.
            vertical_flip: Apply random vertical flip.
            rotation: Maximum rotation angle in degrees.
            crop_size: Size for random crop (int or tuple).
            crop_padding: Padding before random crop.
            resize: Resize to this size.
            random_resized_crop: Size for random resized crop.
            color_jitter: Apply color jitter.
            brightness: Brightness jitter factor.
            contrast: Contrast jitter factor.
            saturation: Saturation jitter factor.
            hue: Hue jitter factor.
            grayscale_prob: Probability of converting to grayscale.
            gaussian_noise: Standard deviation of Gaussian noise.
            gaussian_blur: Apply Gaussian blur.
            blur_kernel_size: Size of blur kernel.
            random_erasing: Apply random erasing.
            cutout: Apply cutout.
            cutout_length: Length of cutout square.
            normalize: Apply normalization.
            mean: Mean for normalization.
            std: Std for normalization.
            randaugment: Use RandAugment.
            randaugment_n: Number of RandAugment operations.
            randaugment_m: Magnitude of RandAugment.
            autoaugment: Use AutoAugment.
            to_tensor: Convert to tensor format.
            channels_first: Output channels first format.
        """
        transforms = []
        
        # Resize first if specified
        if resize is not None:
            transforms.append(Resize(resize))
        
        # Random resized crop
        if random_resized_crop is not None:
            transforms.append(RandomResizedCrop(random_resized_crop))
        
        # Geometric transforms
        if horizontal_flip:
            transforms.append(RandomHorizontalFlip(p=0.5))
        
        if vertical_flip:
            transforms.append(RandomVerticalFlip(p=0.5))
        
        if rotation > 0:
            transforms.append(RandomRotate(max_angle=rotation, p=0.5))
        
        if crop_size is not None:
            transforms.append(RandomCrop(crop_size, padding=crop_padding))
        
        # Color transforms
        if color_jitter or any([brightness, contrast, saturation, hue]):
            transforms.append(ColorJitter(
                brightness=brightness if brightness else (0.2 if color_jitter else 0),
                contrast=contrast if contrast else (0.2 if color_jitter else 0),
                saturation=saturation if saturation else (0.2 if color_jitter else 0),
                hue=hue if hue else (0.1 if color_jitter else 0)
            ))
        
        # Noise and blur
        if gaussian_noise > 0:
            transforms.append(GaussianNoise(std=gaussian_noise, p=0.5))
        
        if gaussian_blur:
            transforms.append(GaussianBlur(kernel_size=blur_kernel_size, p=0.5))
        
        # Advanced augmentation
        if randaugment:
            transforms.append(RandAugment(n=randaugment_n, m=randaugment_m))
        
        if autoaugment:
            transforms.append(AutoAugment())
        
        # Regularization augmentations (applied after other transforms)
        if random_erasing:
            transforms.append(RandomErasing(p=0.5))
        
        if cutout:
            transforms.append(Cutout(length=cutout_length, p=0.5))
        
        # Tensor conversion
        if to_tensor:
            transforms.append(ToTensor(channels_first=channels_first))
        
        # Normalization (should be last)
        if normalize:
            if mean is None:
                mean = [0.485, 0.456, 0.406]  # ImageNet defaults
            if std is None:
                std = [0.229, 0.224, 0.225]
            transforms.append(Normalize(mean=mean, std=std))
        
        self.transform = Compose(transforms) if transforms else None
        
        # Store settings for mixup/cutmix (batch-level augmentations)
        self._mixup = None
        self._cutmix = None
        
    def __call__(self, x):
        """Apply augmentation to input."""
        if self.transform is None:
            return x
        return self.transform(x)
    
    def with_mixup(self, alpha=1.0):
        """Add mixup augmentation (applied at batch level)."""
        self._mixup = Mixup(alpha=alpha)
        return self
    
    def with_cutmix(self, alpha=1.0):
        """Add cutmix augmentation (applied at batch level)."""
        self._cutmix = CutMix(alpha=alpha)
        return self
    
    def augment_batch(self, x, y=None):
        """
        Augment a batch of images with optional mixup/cutmix.
        
        Args:
            x: Batch of images (N, C, H, W) or (N, H, W, C).
            y: Batch of labels.
            
        Returns:
            If y is None: augmented images
            If y is not None and mixup/cutmix enabled: (images, y1, y2, lambda)
            Otherwise: (augmented images, y)
        """
        # Apply per-image transforms
        augmented = np.stack([self(img) for img in x])
        
        # Apply batch-level augmentations
        if y is not None:
            if self._mixup is not None and np.random.random() > 0.5:
                return self._mixup(augmented, y)
            elif self._cutmix is not None and np.random.random() > 0.5:
                return self._cutmix(augmented, y)
            return augmented, y
        
        return augmented
    
    @classmethod
    def for_cifar10(cls, train=True):
        """
        Create augmentor with CIFAR-10 best practices.
        
        Training: RandomCrop(32, padding=4), RandomHorizontalFlip, Normalize
        Evaluation: Normalize only
        """
        if train:
            return cls(
                horizontal_flip=True,
                crop_size=32,
                crop_padding=4,
                normalize=True,
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        else:
            return cls(
                normalize=True,
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
    
    @classmethod
    def for_cifar100(cls, train=True):
        """Create augmentor with CIFAR-100 best practices."""
        if train:
            return cls(
                horizontal_flip=True,
                crop_size=32,
                crop_padding=4,
                normalize=True,
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        else:
            return cls(
                normalize=True,
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
    
    @classmethod
    def for_imagenet(cls, train=True, size=224):
        """
        Create augmentor with ImageNet best practices.
        
        Training: RandomResizedCrop(224), RandomHorizontalFlip, ColorJitter, Normalize
        Evaluation: Resize(256), CenterCrop(224), Normalize
        """
        if train:
            return cls(
                random_resized_crop=size,
                horizontal_flip=True,
                color_jitter=True,
                normalize=True,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:
            return cls(
                resize=256,
                crop_size=size,
                crop_padding=0,  # Center crop, not random
                normalize=True,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
    
    @classmethod
    def for_mnist(cls, train=True):
        """Create augmentor with MNIST best practices."""
        if train:
            return cls(
                rotation=10,
                normalize=True,
                mean=[0.1307],
                std=[0.3081]
            )
        else:
            return cls(
                normalize=True,
                mean=[0.1307],
                std=[0.3081]
            )
    
    @classmethod
    def for_fashion_mnist(cls, train=True):
        """Create augmentor with Fashion-MNIST best practices."""
        if train:
            return cls(
                horizontal_flip=True,
                rotation=10,
                normalize=True,
                mean=[0.2860],
                std=[0.3530]
            )
        else:
            return cls(
                normalize=True,
                mean=[0.2860],
                std=[0.3530]
            )
    
    @classmethod
    def strong(cls, size=224):
        """
        Create augmentor with strong augmentation (for SSL, few-shot, etc.).
        
        Uses RandAugment with additional regularization.
        """
        return cls(
            random_resized_crop=size,
            horizontal_flip=True,
            randaugment=True,
            randaugment_n=2,
            randaugment_m=10,
            random_erasing=True,
            normalize=True
        )
    
    @classmethod
    def light(cls):
        """Create augmentor with light augmentation."""
        return cls(
            horizontal_flip=True,
            brightness=0.1,
            contrast=0.1,
            normalize=True
        )
    
    @classmethod
    def medical(cls, train=True):
        """Create augmentor for medical imaging."""
        if train:
            return cls(
                horizontal_flip=True,
                vertical_flip=True,
                rotation=30,
                brightness=0.1,
                contrast=0.1,
                gaussian_noise=0.02,
                normalize=True,
                mean=[0.5],
                std=[0.5]
            )
        else:
            return cls(
                normalize=True,
                mean=[0.5],
                std=[0.5]
            )
    
    def __repr__(self):
        if self.transform is None:
            return "ImageAugmentor(no transforms)"
        return f"ImageAugmentor(\n{self.transform}\n)"
