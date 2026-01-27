import os
from typing import List, Tuple, Optional

import torch
import random
import numpy as np
from skimage import io
from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import gaussian_blur, elastic_transform
from tqdm import tqdm

from nrrdreader import NRRDReader

def generate_elastic_displacement(
    alpha: float, sigma: float, size: List[int]
) -> torch.Tensor:
    """
    Generate a random elastic displacement field for data augmentation.

    Args:
        alpha (float): Scaling factor for the displacement.
        sigma (float): Standard deviation for Gaussian blur.
        size (List[int]): Size of the displacement field [height, width].

    Returns:
        torch.Tensor: Displacement field of shape (1, H, W, 2).
    """
    alpha = [float(alpha), float(alpha)]
    sigma = [float(sigma), float(sigma)]
    dx = torch.rand([1, 1] + size) * 2 - 1  # Random values in [-1, 1]
    if sigma[0] > 0.0:
        kx = int(8 * sigma[0] + 1)
        if kx % 2 == 0:
            kx += 1  # Kernel size must be odd
        dx = gaussian_blur(dx, [kx, kx], sigma)
    dx = dx * alpha[0] / size[0]

    dy = torch.rand([1, 1] + size) * 2 - 1
    if sigma[1] > 0.0:
        ky = int(8 * sigma[1] + 1)
        if ky % 2 == 0:
            ky += 1
        dy = gaussian_blur(dy, [ky, ky], sigma)
    dy = dy * alpha[1] / size[1]
    # Concatenate dx and dy to get a 2D displacement field
    return torch.concat([dx, dy], 1).permute([0, 2, 3, 1])  # 1 x H x W x 2

def elastic_deformation(
    image: torch.Tensor, displacement: torch.Tensor
) -> torch.Tensor:
    """
    Apply elastic deformation to an image tensor.

    Args:
        image (torch.Tensor): Image tensor (C, H, W).
        displacement (torch.Tensor): Displacement field.

    Returns:
        torch.Tensor: Deformed image tensor.
    """
    tensor = image.unsqueeze(0)  # Add batch dimension
    return elastic_transform(tensor, displacement).squeeze(0)

class ImagePatchBuffer:
    """
    Loads images and labels, extracts patches, and stores them in memory.
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        segment_names: List[str],
        patch_size: int = 1024,
        ignore_label: int = -1
    ):
        """
        Args:
            image_dir (str): Directory containing image files (.tif).
            label_dir (str): Directory containing label files (.nrrd).
            segment_names (List[str]): Names of segments to extract from labels.
            patch_size (int): Size of square patches to extract.
            ignore_label (int): Value to use for unlabeled regions.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.patch_size = patch_size
        self.segment_names = segment_names
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.tif')])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.nrrd')])
        self.num_patches = (2048 / patch_size) * (2048 / patch_size)
        self.ignore_label = ignore_label

        # Ensure image and label files match by name (before extension)
        for img, lbl in zip(self.image_files, self.label_files):
            assert img.split('.')[0] == lbl.split('.')[0], "Mismatch between image and label files"

        assert len(self.image_files) == len(self.label_files), "Mismatch between number of images and labels"

        print(f"Found {len(self.image_files)} image label pairs")



        # Buffers to hold all image and label patches
        self.image_buffer: List[torch.Tensor] = []
        self.label_buffer: List[torch.Tensor] = []
        for img, lbl in tqdm(zip(self.image_files, self.label_files), total=len(self.image_files)):
            image = io.imread(os.path.join(image_dir, img))  # Load image as numpy array
            label = NRRDReader(os.path.join(label_dir, lbl)).extract_segments(self.segment_names)
            label = np.moveaxis(label, -1, 0)  # Move channel axis to first position

            # Mark unlabeled regions with ignore_label
            unlabeled = np.sum(label, axis=0) == 0
            label[:, unlabeled] = self.ignore_label

            # Normalize image to [0, 1]
            image = (image - image.min()) / (image.max() - image.min())

            # Extract patches from image and label
            image_patches, label_patches = self.extract_patches(image, label)
            self.image_buffer.extend([torch.from_numpy(p).float().unsqueeze(0) for p in image_patches])
            self.label_buffer.extend([torch.from_numpy(p) for p in label_patches])

    def extract_patches(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Split image and label into non-overlapping patches.

        Args:
            image (np.ndarray): Image array (H, W).
            label (np.ndarray): Label array (C, H, W).

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Lists of image and label patches.
        """
        image_patches = []
        label_patches = []

        for i in range(0, image.shape[0], self.patch_size):
            for j in range(0, image.shape[1], self.patch_size):
                img_patch = image[i:i + self.patch_size, j:j + self.patch_size]
                lbl_patch = label[:, i:i + self.patch_size, j:j + self.patch_size]
                image_patches.append(img_patch)
                label_patches.append(lbl_patch)

        return image_patches, label_patches

    def get_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.image_buffer[idx], self.label_buffer[idx]

    def __len__(self) -> int:
        return len(self.image_buffer)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_pair(idx)

class ImageSegmentationDataset(Dataset):
    """
    PyTorch Dataset for image segmentation, with optional augmentation.
    """

    def __init__(
        self,
        buffer: ImagePatchBuffer,
        augment: bool = False,
        indices: Optional[List[int]] = None
    ):
        """
        Args:
            buffer (ImagePatchBuffer): Buffer containing image and label patches.
            augment (bool): Whether to apply data augmentation.
            indices (Optional[List[int]]): Optional list of indices to use.
        """
        self.buffer = buffer
        self.should_augment = augment
        self.patch_size = buffer.patch_size
        self.indices = indices

    def __len__(self) -> int:
        """
        Returns:
            int: Number of samples in the dataset.
        """
        if self.indices is not None:
            return len(self.indices)
        else:
            return len(self.buffer)

    def augment(
        self, image: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random augmentations to image and label.

        Args:
            image (torch.Tensor): Image tensor (C, H, W).
            label (torch.Tensor): Label tensor (C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Augmented image and label.
        """
        # Random horizontal flip
        if random.random() > 0.5:
            image = torch.flip(image, [1])
            label = torch.flip(label, [1])

        # Random vertical flip
        if random.random() > 0.5:
            image = torch.flip(image, [2])
            label = torch.flip(label, [2])

        # Random rotation (0, 90, 180, 270 degrees)
        if random.random() > 0.5:
            rots = random.randint(0, 3)
            image = torch.rot90(image, rots, [1, 2])
            label = torch.rot90(label, rots, [1, 2])

        # Elastic deformation (rarely applied)
        if random.random() > 1:
            displacement = generate_elastic_displacement(160, 20, [self.patch_size, self.patch_size])
            image = elastic_deformation(image, displacement)
            label = elastic_deformation(label, displacement)

        return image, label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.indices is not None:
            idx = self.indices[idx]

        img, lbl = self.buffer.get_pair(idx)
        if self.should_augment:
            img, lbl = self.augment(img, lbl)
        return img, lbl
