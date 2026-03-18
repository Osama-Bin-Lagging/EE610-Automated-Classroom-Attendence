"""
augment.py - Data augmentation pipeline for face images.
Generates augmented versions of raw student images to increase training data.
"""

import os
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import random


def horizontal_flip(img):
    """Mirror the image horizontally."""
    return ImageOps.mirror(img)


def adjust_brightness(img, factor_range=(0.8, 1.2)):
    """Randomly adjust brightness by +/- 20%."""
    factor = random.uniform(*factor_range)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def random_rotation(img, max_degrees=10):
    """Rotate image by a random angle within [-max_degrees, max_degrees]."""
    angle = random.uniform(-max_degrees, max_degrees)
    return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(128, 128, 128))


def add_gaussian_noise(img, std=5):
    """Add mild Gaussian noise to the image."""
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def augment_image(img, augmentations=None):
    """
    Apply a random combination of augmentations to an image.

    Args:
        img: PIL Image (RGB)
        augmentations: list of augmentation names to apply
            Options: "flip", "brightness", "rotation", "noise"
            Default: all augmentations

    Returns:
        Augmented PIL Image
    """
    if augmentations is None:
        augmentations = ["flip", "brightness", "rotation", "noise"]

    aug_map = {
        "flip": horizontal_flip,
        "brightness": adjust_brightness,
        "rotation": random_rotation,
        "noise": add_gaussian_noise,
    }

    result = img.copy()
    for aug_name in augmentations:
        if aug_name in aug_map and random.random() > 0.3:  # 70% chance each
            result = aug_map[aug_name](result)

    return result


def generate_augmented_images(img, n_augmented=2):
    """
    Generate n augmented versions of an image.
    Each version gets a random subset of augmentations.

    Args:
        img: PIL Image (RGB)
        n_augmented: number of augmented versions to generate

    Returns:
        List of augmented PIL Images
    """
    augmented = []
    all_augs = ["flip", "brightness", "rotation", "noise"]

    for _ in range(n_augmented):
        # Each augmented image gets a random subset
        result = img.copy()
        for aug_name in all_augs:
            if random.random() > 0.4:  # 60% chance each aug is applied
                if aug_name == "flip":
                    result = horizontal_flip(result)
                elif aug_name == "brightness":
                    result = adjust_brightness(result)
                elif aug_name == "rotation":
                    result = random_rotation(result)
                elif aug_name == "noise":
                    result = add_gaussian_noise(result)
        augmented.append(result)

    return augmented


if __name__ == "__main__":
    # Demo: augment a single image
    import sys
    if len(sys.argv) < 2:
        print("Usage: python augment.py <image_path>")
        sys.exit(1)

    img = Image.open(sys.argv[1])
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")

    augmented = generate_augmented_images(img, n_augmented=3)
    for i, aug in enumerate(augmented):
        out_path = f"augmented_{i+1}.jpg"
        aug.save(out_path)
        print(f"Saved {out_path}")
