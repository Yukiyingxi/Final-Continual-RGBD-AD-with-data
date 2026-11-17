from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff
import torch
from PIL import Image
from torchvision import transforms


def build_transforms(image_size: int, crop_size: int):
    image_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.CenterCrop(crop_size),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    mask_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ]
    )
    return image_transform, mask_transform


class MVTec3DXYZDataset(torch.utils.data.Dataset):
    """Official Dinomaly2 RGB-D input over the processed XYZ tree.

    The official second preprocessing stage performs three 3x3 gap-filling
    rounds and stores only Z.  We reproduce that operation online, avoiding a
    duplicate Z tree while preserving the official HOT rendering exactly.
    """

    def __init__(
        self,
        root: str,
        category: str,
        phase: str,
        image_transform,
        mask_transform,
        class_index: int,
    ) -> None:
        if phase not in {"train", "test"}:
            raise ValueError(f"phase must be train or test, got {phase!r}")
        self.root = Path(root) / category
        self.category = category
        self.phase = phase
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.class_index = int(class_index)
        self.samples = self._discover()

    def _discover(self):
        split = self.root / self.phase
        samples = []
        for defect_dir in sorted(path for path in split.iterdir() if path.is_dir()):
            defect = defect_dir.name
            is_good = defect in {"good", "ok"}
            for rgb_path in sorted((defect_dir / "rgb").glob("*.png")):
                xyz_path = defect_dir / "xyz" / f"{rgb_path.stem}.tiff"
                if not xyz_path.is_file():
                    xyz_path = defect_dir / "z" / f"{rgb_path.stem}.tiff"
                if not xyz_path.is_file():
                    raise FileNotFoundError(f"Missing XYZ/Z TIFF for {rgb_path}")
                gt_path = None if is_good else defect_dir / "gt" / rgb_path.name
                if gt_path is not None and not gt_path.is_file():
                    raise FileNotFoundError(f"Missing mask for {rgb_path}: {gt_path}")
                samples.append((rgb_path, xyz_path, gt_path, 0 if is_good else 1, defect))
        if not samples:
            raise RuntimeError(f"No samples found under {split}")
        return samples

    @staticmethod
    def _fill_depth_gaps(depth: np.ndarray, rounds: int = 3) -> np.ndarray:
        depth = np.nan_to_num(
            depth, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32, copy=True)
        kernel = np.ones((3, 3), dtype=np.float32)
        for _ in range(rounds):
            valid = depth > 0.0
            neighbor_sum = cv2.filter2D(
                depth, -1, kernel, borderType=cv2.BORDER_CONSTANT
            )
            neighbor_count = cv2.filter2D(
                valid.astype(np.float32),
                -1,
                kernel,
                borderType=cv2.BORDER_CONSTANT,
            )
            fillable = (~valid) & (neighbor_count > 0.0)
            # The reference sliced neighborhoods leave top and left borders.
            fillable[0, :] = False
            fillable[:, 0] = False
            depth[fillable] = neighbor_sum[fillable] / neighbor_count[fillable]
        return depth

    @staticmethod
    def _render_depth(xyz_path: Path):
        xyz = tiff.imread(str(xyz_path))
        if xyz.ndim == 3:
            if xyz.shape[-1] < 3:
                raise ValueError(f"Expected XYZ channels in {xyz_path}, got {xyz.shape}")
            depth = xyz[..., 2]
        elif xyz.ndim == 2:
            depth = xyz
        else:
            raise ValueError(f"Unsupported TIFF shape {xyz.shape} in {xyz_path}")

        depth = MVTec3DXYZDataset._fill_depth_gaps(depth, rounds=3)
        valid = depth > 0.0
        rendered = np.zeros_like(depth, dtype=np.float32)
        values = depth[valid]
        if values.size:
            lo, hi = float(values.min()), float(values.max())
            rendered[valid] = 1.0 - 0.8 * (
                (values - lo) / max(hi - lo, 1e-8)
            )
        heatmap = cv2.applyColorMap(
            np.clip(rendered * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_HOT
        )
        return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), valid

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        rgb_path, xyz_path, gt_path, anomaly_label, _ = self.samples[index]
        rgb = np.asarray(Image.open(rgb_path).convert("RGB")).copy()
        depth_rgb, valid = self._render_depth(xyz_path)
        rgb[~valid] = 0

        rgb_tensor = self.image_transform(Image.fromarray(rgb))
        depth_tensor = self.image_transform(Image.fromarray(depth_rgb))
        image = torch.cat([rgb_tensor, depth_tensor], dim=0)
        if gt_path is None:
            mask = torch.zeros(
                (1, image.shape[-2], image.shape[-1]), dtype=torch.float32
            )
        else:
            # Official evaluation calls gt.bool(), so every non-zero pixel
            # introduced by PIL resizing belongs to the binary region.
            mask = (self.mask_transform(Image.open(gt_path).convert("L")) > 0.0).float()
        return image, mask, int(anomaly_label), str(rgb_path), self.class_index


def build_category_datasets(
    root: str, categories, phase: str, image_size: int, crop_size: int
):
    image_transform, mask_transform = build_transforms(image_size, crop_size)
    return [
        MVTec3DXYZDataset(
            root=root,
            category=category,
            phase=phase,
            image_transform=image_transform,
            mask_transform=mask_transform,
            class_index=index,
        )
        for index, category in enumerate(categories)
    ]
