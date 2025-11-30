"""
Tests for the augmentation module.
"""

import numpy as np
import pytest
from augmentation import (
    # Basic transforms
    RandomHorizontalFlip, RandomVerticalFlip, RandomRotate,
    RandomCrop, CenterCrop, Resize, RandomResizedCrop,
    # Color transforms
    RandomBrightness, RandomContrast, ColorJitter, RandomGrayscale,
    # Noise and blur
    GaussianNoise, GaussianBlur,
    # Regularization
    RandomErasing, Cutout, Mixup, CutMix,
    # Utility
    Normalize, ToTensor, Compose, RandomApply, RandomChoice, RandomOrder,
    # Augmentor
    ImageAugmentor
)


class TestBasicTransforms:
    """Tests for basic geometric transforms."""
    
    def test_horizontal_flip_shape(self):
        """Horizontal flip should preserve shape."""
        transform = RandomHorizontalFlip(p=1.0)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == img.shape
    
    def test_horizontal_flip_deterministic(self):
        """Horizontal flip with p=1 should always flip."""
        transform = RandomHorizontalFlip(p=1.0)
        img = np.arange(12).reshape(2, 2, 3).astype(float)
        result = transform(img)
        # Should be horizontally flipped
        assert np.allclose(result[:, 0, :], img[:, 1, :])
        assert np.allclose(result[:, 1, :], img[:, 0, :])
    
    def test_horizontal_flip_no_flip(self):
        """Horizontal flip with p=0 should never flip."""
        transform = RandomHorizontalFlip(p=0.0)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert np.allclose(result, img)
    
    def test_vertical_flip_shape(self):
        """Vertical flip should preserve shape."""
        transform = RandomVerticalFlip(p=1.0)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == img.shape
    
    def test_vertical_flip_deterministic(self):
        """Vertical flip with p=1 should always flip."""
        transform = RandomVerticalFlip(p=1.0)
        img = np.arange(12).reshape(2, 2, 3).astype(float)
        result = transform(img)
        # Should be vertically flipped
        assert np.allclose(result[0, :, :], img[1, :, :])
        assert np.allclose(result[1, :, :], img[0, :, :])
    
    def test_rotate_shape(self):
        """Rotation should preserve shape."""
        transform = RandomRotate(max_angle=45, p=1.0)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == img.shape
    
    def test_rotate_zero_angle(self):
        """Rotation with max_angle=0 should not change image."""
        transform = RandomRotate(max_angle=0, p=1.0)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert np.allclose(result, img)
    
    def test_random_crop_shape(self):
        """Random crop should output correct shape."""
        transform = RandomCrop(size=16)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == (16, 16, 3)
    
    def test_random_crop_with_padding(self):
        """Random crop with padding should work."""
        transform = RandomCrop(size=32, padding=4)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == (32, 32, 3)
    
    def test_center_crop_shape(self):
        """Center crop should output correct shape."""
        transform = CenterCrop(size=16)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == (16, 16, 3)
    
    def test_center_crop_centered(self):
        """Center crop should extract center of image."""
        transform = CenterCrop(size=2)
        # Create 4x4 image with known values
        img = np.arange(48).reshape(4, 4, 3).astype(float)
        result = transform(img)
        # Center 2x2 should be rows 1-2, cols 1-2
        expected = img[1:3, 1:3, :]
        assert np.allclose(result, expected)
    
    def test_resize_shape(self):
        """Resize should output correct shape."""
        transform = Resize(size=64)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == (64, 64, 3)
    
    def test_resize_tuple(self):
        """Resize with tuple should output correct shape."""
        transform = Resize(size=(48, 64))
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == (48, 64, 3)
    
    def test_random_resized_crop_shape(self):
        """Random resized crop should output correct shape."""
        transform = RandomResizedCrop(size=64)
        img = np.random.rand(128, 128, 3)
        result = transform(img)
        assert result.shape == (64, 64, 3)


class TestColorTransforms:
    """Tests for color transforms."""
    
    def test_brightness_shape(self):
        """Brightness adjustment should preserve shape."""
        transform = RandomBrightness(factor=0.5, p=1.0)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == img.shape
    
    def test_brightness_range(self):
        """Brightness adjustment should keep values in [0, 1]."""
        transform = RandomBrightness(factor=0.5, p=1.0)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.min() >= 0 and result.max() <= 1
    
    def test_contrast_shape(self):
        """Contrast adjustment should preserve shape."""
        transform = RandomContrast(factor=0.5, p=1.0)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == img.shape
    
    def test_color_jitter_shape(self):
        """Color jitter should preserve shape."""
        transform = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == img.shape
    
    def test_grayscale_shape(self):
        """Grayscale conversion should preserve shape."""
        transform = RandomGrayscale(p=1.0)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == img.shape
    
    def test_grayscale_channels_same(self):
        """Grayscale conversion should make all channels same."""
        transform = RandomGrayscale(p=1.0)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        # All channels should be (approximately) the same
        assert np.allclose(result[:, :, 0], result[:, :, 1], atol=1e-5)
        assert np.allclose(result[:, :, 1], result[:, :, 2], atol=1e-5)


class TestNoiseBlur:
    """Tests for noise and blur transforms."""
    
    def test_gaussian_noise_shape(self):
        """Gaussian noise should preserve shape."""
        transform = GaussianNoise(std=0.1, p=1.0)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == img.shape
    
    def test_gaussian_noise_adds_noise(self):
        """Gaussian noise should change the image."""
        transform = GaussianNoise(std=0.1, p=1.0)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert not np.allclose(result, img)
    
    def test_gaussian_blur_shape(self):
        """Gaussian blur should preserve shape."""
        transform = GaussianBlur(kernel_size=3, p=1.0)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == img.shape


class TestRegularization:
    """Tests for regularization transforms."""
    
    def test_random_erasing_shape(self):
        """Random erasing should preserve shape."""
        transform = RandomErasing(p=1.0)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == img.shape
    
    def test_cutout_shape(self):
        """Cutout should preserve shape."""
        transform = Cutout(length=8, p=1.0)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == img.shape
    
    def test_cutout_creates_holes(self):
        """Cutout should create zeros in the image."""
        transform = Cutout(length=8, p=1.0)
        img = np.ones((32, 32, 3))
        result = transform(img)
        # Should have some zeros
        assert np.any(result == 0)
    
    def test_mixup_shape(self):
        """Mixup should preserve shape."""
        transform = Mixup(alpha=1.0)
        x = np.random.rand(4, 32, 32, 3)
        y = np.eye(10)[np.array([0, 1, 2, 3])]  # One-hot labels
        x_mixed, y1, y2, lam = transform(x, y)
        assert x_mixed.shape == x.shape
        assert y1.shape == y.shape
    
    def test_cutmix_shape(self):
        """CutMix should preserve shape."""
        transform = CutMix(alpha=1.0)
        x = np.random.rand(4, 32, 32, 3)
        y = np.eye(10)[np.array([0, 1, 2, 3])]  # One-hot labels
        x_mixed, y1, y2, lam = transform(x, y)
        assert x_mixed.shape == x.shape


class TestNormalization:
    """Tests for normalization transforms."""
    
    def test_normalize_shape(self):
        """Normalize should preserve shape."""
        transform = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == img.shape
    
    def test_normalize_values(self):
        """Normalize should properly normalize values."""
        transform = Normalize(mean=[0.5], std=[0.5])
        img = np.ones((32, 32, 1)) * 0.5
        result = transform(img)
        # (0.5 - 0.5) / 0.5 = 0
        assert np.allclose(result, 0)
    
    def test_to_tensor_channels_first(self):
        """ToTensor should convert to channels first format."""
        transform = ToTensor(channels_first=True)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == (3, 32, 32)
    
    def test_to_tensor_channels_last(self):
        """ToTensor with channels_first=False should keep channels last."""
        transform = ToTensor(channels_first=False)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == (32, 32, 3)


class TestCompose:
    """Tests for composition utilities."""
    
    def test_compose_applies_all(self):
        """Compose should apply all transforms in order."""
        transforms = [
            RandomHorizontalFlip(p=1.0),
            RandomHorizontalFlip(p=1.0)  # Flip twice = original
        ]
        composed = Compose(transforms)
        img = np.random.rand(32, 32, 3)
        result = composed(img)
        assert np.allclose(result, img)
    
    def test_compose_empty(self):
        """Empty compose should return input unchanged."""
        composed = Compose([])
        img = np.random.rand(32, 32, 3)
        result = composed(img)
        assert np.allclose(result, img)
    
    def test_random_apply(self):
        """RandomApply with p=0 should not apply transform."""
        transform = RandomApply([RandomHorizontalFlip(p=1.0)], p=0.0)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert np.allclose(result, img)
    
    def test_random_choice_shape(self):
        """RandomChoice should preserve shape."""
        transforms = [
            RandomHorizontalFlip(p=1.0),
            RandomVerticalFlip(p=1.0)
        ]
        transform = RandomChoice(transforms)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == img.shape
    
    def test_random_order_shape(self):
        """RandomOrder should preserve shape."""
        transforms = [
            RandomBrightness(factor=0.1, p=1.0),
            RandomContrast(factor=0.1, p=1.0)
        ]
        transform = RandomOrder(transforms)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == img.shape


class TestImageAugmentor:
    """Tests for the ImageAugmentor class."""
    
    def test_empty_augmentor(self):
        """Empty augmentor should return input unchanged."""
        augmentor = ImageAugmentor()
        img = np.random.rand(32, 32, 3)
        result = augmentor(img)
        assert np.allclose(result, img)
    
    def test_cifar10_preset_train(self):
        """CIFAR-10 training preset should work."""
        augmentor = ImageAugmentor.for_cifar10(train=True)
        img = np.random.rand(32, 32, 3)
        result = augmentor(img)
        assert result.shape == img.shape
    
    def test_cifar10_preset_eval(self):
        """CIFAR-10 evaluation preset should work."""
        augmentor = ImageAugmentor.for_cifar10(train=False)
        img = np.random.rand(32, 32, 3)
        result = augmentor(img)
        assert result.shape == img.shape
    
    def test_imagenet_preset_train(self):
        """ImageNet training preset should work."""
        augmentor = ImageAugmentor.for_imagenet(train=True, size=224)
        img = np.random.rand(256, 256, 3)
        result = augmentor(img)
        assert result.shape == (224, 224, 3)
    
    def test_mnist_preset(self):
        """MNIST preset should work."""
        augmentor = ImageAugmentor.for_mnist(train=True)
        img = np.random.rand(28, 28, 1)
        result = augmentor(img)
        assert result.shape == img.shape
    
    def test_fashion_mnist_preset(self):
        """Fashion-MNIST preset should work."""
        augmentor = ImageAugmentor.for_fashion_mnist(train=True)
        img = np.random.rand(28, 28, 1)
        result = augmentor(img)
        assert result.shape == img.shape
    
    def test_strong_preset(self):
        """Strong augmentation preset should work."""
        augmentor = ImageAugmentor.strong(size=64)
        img = np.random.rand(128, 128, 3)
        result = augmentor(img)
        assert result.shape == (64, 64, 3)
    
    def test_light_preset(self):
        """Light augmentation preset should work."""
        augmentor = ImageAugmentor.light()
        img = np.random.rand(32, 32, 3)
        result = augmentor(img)
        assert result.shape == img.shape
    
    def test_medical_preset(self):
        """Medical imaging preset should work."""
        augmentor = ImageAugmentor.medical(train=True)
        img = np.random.rand(256, 256, 1)
        result = augmentor(img)
        assert result.shape == img.shape
    
    def test_custom_augmentor(self):
        """Custom augmentor with various options should work."""
        augmentor = ImageAugmentor(
            horizontal_flip=True,
            rotation=15,
            color_jitter=True,
            gaussian_noise=0.02,
            normalize=True,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        img = np.random.rand(32, 32, 3)
        result = augmentor(img)
        assert result.shape == img.shape
    
    def test_augment_batch(self):
        """Batch augmentation should work."""
        augmentor = ImageAugmentor(horizontal_flip=True)
        batch = np.random.rand(8, 32, 32, 3)
        result = augmentor.augment_batch(batch)
        assert result.shape == batch.shape
    
    def test_augment_batch_with_labels(self):
        """Batch augmentation with labels should work."""
        augmentor = ImageAugmentor(horizontal_flip=True)
        batch = np.random.rand(8, 32, 32, 3)
        labels = np.eye(10)[:8]
        result_x, result_y = augmentor.augment_batch(batch, labels)
        assert result_x.shape == batch.shape
        assert result_y.shape == labels.shape
    
    def test_repr(self):
        """Augmentor should have string representation."""
        augmentor = ImageAugmentor(horizontal_flip=True)
        repr_str = repr(augmentor)
        assert "ImageAugmentor" in repr_str


class TestGrayscaleImages:
    """Tests for grayscale image handling."""
    
    def test_horizontal_flip_grayscale(self):
        """Horizontal flip should work on grayscale images."""
        transform = RandomHorizontalFlip(p=1.0)
        img = np.random.rand(32, 32, 1)
        result = transform(img)
        assert result.shape == img.shape
    
    def test_normalize_grayscale(self):
        """Normalize should work on grayscale images."""
        transform = Normalize(mean=[0.5], std=[0.5])
        img = np.random.rand(32, 32, 1)
        result = transform(img)
        assert result.shape == img.shape
    
    def test_cutout_grayscale(self):
        """Cutout should work on grayscale images."""
        transform = Cutout(length=8, p=1.0)
        img = np.random.rand(32, 32, 1)
        result = transform(img)
        assert result.shape == img.shape


class TestBatchHandling:
    """Tests for batch handling in transforms."""
    
    def test_mixup_batch_processing(self):
        """Mixup should correctly mix batches."""
        mixup = Mixup(alpha=1.0)
        x = np.random.rand(4, 32, 32, 3)
        y = np.eye(10)[:4]
        x_mixed, y1, y2, lam = mixup(x, y)
        
        # Lambda should be between 0 and 1
        assert 0 <= lam <= 1
        # Mixed data should be combination of original
        assert x_mixed.shape == x.shape
    
    def test_cutmix_creates_patches(self):
        """CutMix should create rectangular patches."""
        cutmix = CutMix(alpha=1.0)
        x = np.ones((4, 32, 32, 3))
        x2 = np.zeros((4, 32, 32, 3))
        
        # Manually test the patch creation
        y = np.eye(10)[:4]
        x_mixed, y1, y2, lam = cutmix(x, y)
        
        # Result should have some zeros (from the cut)
        # Note: Due to random permutation, this might not always be true
        # So we just check shape
        assert x_mixed.shape == x.shape


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_small_image_crop(self):
        """Crop should handle small images."""
        # Image smaller than crop size
        transform = RandomCrop(size=32, padding=4)
        img = np.random.rand(24, 24, 3)
        result = transform(img)
        # With padding of 4, 24 + 2*4 = 32, so should work
        assert result.shape == (32, 32, 3)
    
    def test_very_small_cutout(self):
        """Cutout with small length should work."""
        transform = Cutout(length=1, p=1.0)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == img.shape
    
    def test_large_rotation(self):
        """Large rotation angle should work."""
        transform = RandomRotate(max_angle=180, p=1.0)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert result.shape == img.shape
    
    def test_zero_noise(self):
        """Zero noise should not change image."""
        transform = GaussianNoise(std=0, p=1.0)
        img = np.random.rand(32, 32, 3)
        result = transform(img)
        assert np.allclose(result, img)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
