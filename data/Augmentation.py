import os
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image, ImageDraw
import numpy as np
from timm.data import create_transform
import pandas as pd
from sklearn.model_selection import train_test_split
import pydicom
import sys
import ipdb
import albumentations
import albumentations.augmentations.transforms as transforms
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import pandas as pd
from torchvision import transforms
import albumentations as A
import random
from typing import Any, List, Optional, Tuple




class RealisticMetalArtifact(A.ImageOnlyTransform):
    """
    金属伪影（CT/DR风格）仿真：
    - 不规则椭圆金属形状（轻微形变/噪声）
    - 射线条纹：随机角度/强度/宽度 + 径向衰减 + 模糊
    - 金属与背景融合 + 少量高斯噪声
    支持强度级别 level ∈ {1,2,3,4} 和多金属 num_masks
    """

    def __init__(
        self,
        level: int = 1,
        num_masks: int = 1,
        base_radius: int = 34,
        line_count_range: Tuple[int, int] = (10, 26),
        intensity_range: Tuple[float, float] = (0.20, 0.55),
        width_range: Tuple[int, int] = (1, 4),
        ray_length_scale: float = 2.8,
        radial_decay: float = 2.2,
        blur_sigma: float = 1.3,
        blend_alpha: float = 0.55,
        noise_std_base: float = 0.02,
        seed: Optional[int] = None,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.level = int(np.clip(level, 1, 4))
        self.num_masks = int(max(1, num_masks))
        self.base_radius = int(max(4, base_radius))
        self.line_count_range = tuple(line_count_range)
        self.intensity_range = tuple(intensity_range)
        self.width_range = tuple(width_range)
        self.ray_length_scale = float(ray_length_scale)
        self.radial_decay = float(radial_decay)
        self.blur_sigma = float(blur_sigma)
        self.blend_alpha = float(np.clip(blend_alpha, 0.0, 1.0))
        self.noise_std_base = float(noise_std_base)
        self.seed = seed

    # 便于 Albumentations 存/回放
    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return (
            "level",
            "num_masks",
            "base_radius",
            "line_count_range",
            "intensity_range",
            "width_range",
            "ray_length_scale",
            "radial_decay",
            "blur_sigma",
            "blend_alpha",
            "noise_std_base",
            "seed",
        )

    @staticmethod
    def _value_max_for_dtype(img: np.ndarray) -> float:
        if np.issubdtype(img.dtype, np.integer):
            return float(np.iinfo(img.dtype).max)
        vmax = float(np.max(img)) if img.size else 1.0
        return 1.0 if vmax <= 1.5 else vmax

    @staticmethod
    def _odd_ge(n: float, minv: int = 3) -> int:
        k = max(minv, int(n))
        return k if (k % 2 == 1) else (k + 1)

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        is_hwc = (img.ndim == 3 and img.shape[2] > 1)
        orig_dtype = img.dtype
        rng = np.random.default_rng(self.seed)  # numpy >=1.17

        if is_hwc:
            rays_cache, metal_mask_cache = None, None
            out_channels = []
            for c in range(img.shape[2]):
                ch = img[..., c]
                out_ch, rays_cache, metal_mask_cache = self._apply_single_channel(
                    ch, rng, rays_cache=rays_cache, metal_mask_cache=metal_mask_cache
                )
                out_channels.append(out_ch.astype(orig_dtype, copy=False))
            out = np.stack(out_channels, axis=-1)
        else:
            out, _, _ = self._apply_single_channel(img, rng)
            out = out.astype(orig_dtype, copy=False)
        return out

    def _apply_single_channel(
        self,
        image: np.ndarray,
        rng: Any,  # np.random.Generator（为兼容老 numpy 这里用 Any）
        rays_cache: Optional[np.ndarray] = None,
        metal_mask_cache: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img = image.astype(np.float32, copy=True)
        H, W = img.shape[:2]
        vmax = self._value_max_for_dtype(image)

        # 按 level 调参 + 小图保护
        level = self.level
        max_rad_by_size = max(4, (min(H, W) - 2) // 3)
        radius = int(min(self.base_radius + (level - 1) * 6, max_rad_by_size))

        n_lines = int(rng.integers(self.line_count_range[0], self.line_count_range[1] + 1))
        ray_amp = vmax * (0.10 + 0.05 * (level - 1))
        metal_val = vmax * (0.88 + 0.03 * level)
        noise_std = vmax * self.noise_std_base * (0.8 + 0.2 * level)

        # 1) 金属 mask
        if metal_mask_cache is None:
            metal_mask = np.zeros((H, W), dtype=np.uint8)
            centers: List[Tuple[int, int]] = []

            safe_l = radius
            safe_rx = max(radius + 1, W - radius)
            safe_ry = max(radius + 1, H - radius)

            for _ in range(self.num_masks):
                if safe_rx <= safe_l or safe_ry <= safe_l:
                    cx, cy = W // 2, H // 2
                else:
                    cx = int(rng.integers(safe_l, safe_rx))
                    cy = int(rng.integers(safe_l, safe_ry))
                centers.append((cx, cy))

                ax = int(radius * rng.uniform(0.8, 1.25))
                ay = int(radius * rng.uniform(0.6, 1.1))
                angle = float(rng.uniform(0, 180))
                cv2.ellipse(metal_mask, (cx, cy), (ax, ay), angle, 0, 360, 255, -1)

                k = self._odd_ge(radius * 0.08, 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_OPEN, kernel)

                # 轻微边界抖动
                jitter = (rng.random((H, W)) * 255).astype(np.uint8)
                metal_mask = cv2.addWeighted(metal_mask, 0.9, jitter, 0.1, 0)
                _, metal_mask = cv2.threshold(metal_mask, 127, 255, cv2.THRESH_BINARY)
        else:
            metal_mask = metal_mask_cache.copy()
            centers = self._extract_centers_from_mask(metal_mask)

        # 2) 提亮 & 融合
        metal_layer = img.copy()
        metal_layer[metal_mask > 0] = metal_val
        out = cv2.addWeighted(img, 1.0 - self.blend_alpha, metal_layer, self.blend_alpha, 0)

        # 3) 射线条纹
        if rays_cache is None:
            rays = np.zeros_like(out, dtype=np.float32)
            yy, xx = np.indices((H, W))
            L = int(radius * self.ray_length_scale * (1.0 + 0.15 * (level - 1)))

            for (cx, cy) in centers:
                base_angles = np.linspace(0, 2 * np.pi, n_lines, endpoint=False)
                angles = base_angles + rng.normal(0, 0.08, size=base_angles.shape)

                for theta in angles:
                    width = int(rng.integers(self.width_range[0], self.width_range[1] + 1))
                    inten = float(rng.uniform(self.intensity_range[0], self.intensity_range[1]))
                    inten *= (1.0 + 0.12 * (level - 1)) * vmax

                    x2 = int(np.clip(cx + np.cos(theta) * L, 0, W - 1))
                    y2 = int(np.clip(cy + np.sin(theta) * L, 0, H - 1))

                    ray_map = np.zeros_like(out, dtype=np.float32)
                    cv2.line(ray_map, (int(cx), int(cy)), (x2, y2), float(inten), int(width))

                    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2, dtype=np.float32)
                    decay = np.exp(-dist / max(1e-6, radius * self.radial_decay)).astype(np.float32)
                    ray_map *= decay

                    ksize = self._odd_ge(width * 3, 3)
                    ray_map = cv2.GaussianBlur(ray_map, (ksize, ksize), self.blur_sigma)

                    rays += ray_map

            m = float(np.max(rays))
            if m > 0:
                rays = rays / m * ray_amp
        else:
            rays = rays_cache

        out = np.clip(out + rays, 0, vmax).astype(np.float32)

        # 4) 邻域平滑，弱化贴片感
        dil_k = self._odd_ge(radius * 0.2, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_k, dil_k))
        neigh = cv2.dilate(metal_mask, kernel)
        smooth = cv2.GaussianBlur(out, (0, 0), sigmaX=1.4 + 0.3 * level)
        out = np.where(neigh > 0, smooth, out)

        # 5) 噪声（同一 rng，便于复现）
        noise = rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)
        out = np.clip(out + noise, 0, vmax).astype(np.float32)

        return out, rays, metal_mask

    @staticmethod
    def _extract_centers_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
        centers: List[Tuple[int, int]] = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) < 3:
                continue
            M = cv2.moments(cnt)
            if abs(M.get("m00", 0.0)) < 1e-6:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
            else:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
        if not centers:
            h, w = mask.shape[:2]
            centers = [(w // 2, h // 2)]
        return centers
        
class CustomMetalArtifact(A.ImageOnlyTransform):
    def __init__(self, level=1, always_apply=False, p=1.0, metal_value=255, num_masks=1,
                 add_star_artifact=True, num_lines=18, line_intensity=90, line_width=2):
        super().__init__(always_apply, p)
        self.level = max(1, min(4, level))
        self.level_to_radius = {
            1: 36,    # small
            2: 40,
            3: 44,
            4: 48,
        }
        self.metal_value = metal_value
        self.num_masks = num_masks
        self.add_star_artifact = add_star_artifact
        self.num_lines = num_lines
        self.line_intensity = line_intensity
        self.line_width = line_width

    def apply(self, img, **params):
        # 只支持单通道（灰度），如需多通道可拓展
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        radius = self.level_to_radius[self.level]
        h, w = img.shape[:2]
        rng = np.random.default_rng()
        centers = []
        for _ in range(self.num_masks):
            cx = rng.integers(radius, w-radius)
            cy = rng.integers(radius, h-radius)
            centers.append((cx, cy))
            cv2.circle(mask, (cx, cy), radius, 1, -1)
        # 覆盖金属区域
        out_img = img.copy()
        out_img[mask == 1] = self.metal_value

        # 可选：叠加star artifact射线伪影
        if self.add_star_artifact:
            for (cx, cy) in centers:
                for i in range(self.num_lines):
                    angle = 2 * np.pi * i / self.num_lines
                    dx = int(np.cos(angle) * radius * 2.2)   # 射线长度略大于金属
                    dy = int(np.sin(angle) * radius * 2.2)
                    pt1 = (int(cx), int(cy))
                    pt2 = (int(cx + dx), int(cy + dy))
                    cv2.line(out_img, pt1, pt2, color=int(self.line_intensity), thickness=self.line_width)
        return out_img.astype(img.dtype)


class CustomBrightnessEnhance(A.ImageOnlyTransform):
    def __init__(self, level=0, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        # level 范围: -2 到 +2，0 表示无变化
        self.level = max(-2, min(2, level))  # 限制范围
    
    def apply(self, img, **params):
        from PIL import Image, ImageEnhance
        import numpy as np
        
        pil_img = Image.fromarray(img)
        
        # level = 0 时不添加亮度变化
        if self.level == 0:
            factor = 1.0  # 无变化
        else:
            # 将 -2 到 +2 的范围映射到合适的亮度因子
            # 负值降低亮度，正值增加亮度
            if self.level < 0:
                # -2 到 0: 映射到 0.3 到 1.0 (降低亮度)
                factor = 0.3 + (self.level + 2) * 0.35
            else:
                # 0 到 2: 映射到 1.0 到 1.8 (增加亮度)
                factor = 1.0 + self.level * 0.4
        
        # 添加少量随机噪声
        noisy_factor = factor + np.random.uniform(-0.01, 0.01)
        
        enhanced = ImageEnhance.Brightness(pil_img).enhance(noisy_factor)
        return np.array(enhanced)

class CustomGaussianNoise(A.ImageOnlyTransform):
    """
    对原始CT图像施加高斯噪声，level=1~4
    σ分别为5, 10, 15, 20，适合uint8图像。
    """
    def __init__(self, level=1, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.level = max(1, min(4, level))
        self.level_to_std = {
            1: 5,
            2: 10,
            3: 15,
            4: 20
        }
    
    def apply(self, img, **params):
        import numpy as np
        std = self.level_to_std[self.level]
        noise = np.random.normal(0, std, img.shape)
        noisy_img = img.astype(np.float32) + noise
        noisy_img = np.clip(noisy_img, 0, 255)
        return noisy_img.astype(img.dtype)

class CustomGammaCorrection(A.ImageOnlyTransform):
    """
    分级伽马校正（全部变暗），参考医学鲁棒研究常用设置。
    level=1~4，对应 gamma=[1.2,1.5,1.8,2.2]
    """
    def __init__(self, level=1, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.level = max(1, min(4, level))
        self.level_to_gamma = {
            1: 1.2,  # 轻度变暗
            2: 1.5,  # 中度变暗
            3: 1.8,  # 强变暗
            4: 2.2   # 极强变暗
        }

    def apply(self, img, **params):
        import numpy as np
        gamma = self.level_to_gamma[self.level]
        img = img.astype(np.float32) / 255.0
        img = np.power(img, gamma)
        img = np.clip(img * 255.0, 0, 255)
        return img.astype(np.uint8)