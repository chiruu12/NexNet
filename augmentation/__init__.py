"""
Data Augmentation Module for NexNet.

Provides comprehensive image augmentation techniques for training
neural networks with enhanced generalization capabilities.

Usage:
    from augmentation import (
        ImageAugmentor, Compose, RandomFlip, RandomRotate,
        RandomCrop, ColorJitter, Normalize, ToTensor
    )
    
    # Create augmentation pipeline
    augmentor = Compose([
        RandomFlip(horizontal=True),
        RandomRotate(max_angle=15),
        ColorJitter(brightness=0.2, contrast=0.2),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply to batch
    augmented = augmentor(images)
"""

from .transforms import (
    Compose,
    ToTensor,
    Normalize,
    RandomFlip,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotate,
    RandomCrop,
    CenterCrop,
    Resize,
    RandomResizedCrop,
    ColorJitter,
    RandomBrightness,
    RandomContrast,
    RandomSaturation,
    RandomHue,
    GaussianNoise,
    GaussianBlur,
    RandomErasing,
    Cutout,
    Mixup,
    CutMix,
    RandomAffine,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomPerspective,
    RandomGrayscale,
    RandomInvert,
    RandomSolarize,
    RandomPosterize,
    RandomApply,
    RandomChoice,
    RandomOrder,
    AutoAugment,
    RandAugment,
)

from .augmentor import ImageAugmentor

__all__ = [
    'Compose',
    'ToTensor',
    'Normalize',
    'RandomFlip',
    'RandomHorizontalFlip',
    'RandomVerticalFlip',
    'RandomRotate',
    'RandomCrop',
    'CenterCrop',
    'Resize',
    'RandomResizedCrop',
    'ColorJitter',
    'RandomBrightness',
    'RandomContrast',
    'RandomSaturation',
    'RandomHue',
    'GaussianNoise',
    'GaussianBlur',
    'RandomErasing',
    'Cutout',
    'Mixup',
    'CutMix',
    'RandomAffine',
    'ElasticTransform',
    'GridDistortion',
    'OpticalDistortion',
    'RandomPerspective',
    'RandomGrayscale',
    'RandomInvert',
    'RandomSolarize',
    'RandomPosterize',
    'RandomApply',
    'RandomChoice',
    'RandomOrder',
    'AutoAugment',
    'RandAugment',
    'ImageAugmentor',
]
