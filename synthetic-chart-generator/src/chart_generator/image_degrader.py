"""Apply realistic scan/fax degradations to a page image.

Three named profiles (small/medium/complex) preset the intensity. The same
DegradationProfile can be constructed manually for finer control.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image


@dataclass
class DegradationProfile:
    name: str
    blur_kernel: int = 0  # 0 disables; otherwise odd integer >= 3
    gaussian_noise_sigma: float = 0.0  # std-dev in 0-255 space
    rotation_deg: float = 0.0  # max absolute rotation (uniform sample)
    speckle_density: float = 0.0  # fraction of pixels to flip to black/white
    jpeg_quality: int = 95  # post-compression artifact intensity (lower = worse)
    contrast: float = 1.0  # 1.0 keeps original contrast
    brightness: int = 0  # signed offset added before clipping
    horizontal_streaks: int = 0  # fax-style scan-line artifacts
    seed: int | None = None


SMALL_PROFILE = DegradationProfile(
    name="small",
    blur_kernel=3,
    gaussian_noise_sigma=4.0,
    rotation_deg=0.4,
    speckle_density=0.0005,
    jpeg_quality=85,
    contrast=0.97,
)

MEDIUM_PROFILE = DegradationProfile(
    name="medium",
    blur_kernel=3,
    gaussian_noise_sigma=10.0,
    rotation_deg=1.2,
    speckle_density=0.003,
    jpeg_quality=65,
    contrast=0.9,
    brightness=-10,
    horizontal_streaks=2,
)

COMPLEX_PROFILE = DegradationProfile(
    name="complex",
    blur_kernel=5,
    gaussian_noise_sigma=18.0,
    rotation_deg=2.5,
    speckle_density=0.012,
    jpeg_quality=40,
    contrast=0.82,
    brightness=-22,
    horizontal_streaks=6,
)


def _to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("L"))  # grayscale matches scan/fax output
    return arr


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")


def _rotate(arr: np.ndarray, max_deg: float, rng: random.Random) -> np.ndarray:
    if max_deg <= 0:
        return arr
    angle = rng.uniform(-max_deg, max_deg)
    h, w = arr.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        arr,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def _add_speckle(arr: np.ndarray, density: float, rng: random.Random) -> np.ndarray:
    if density <= 0:
        return arr
    out = arr.copy()
    h, w = out.shape
    n = int(density * h * w)
    seed = rng.randrange(2**31)
    np_rng = np.random.default_rng(seed)
    ys = np_rng.integers(0, h, size=n)
    xs = np_rng.integers(0, w, size=n)
    vals = np_rng.choice([0, 255], size=n)
    out[ys, xs] = vals
    return out


def _add_streaks(arr: np.ndarray, count: int, rng: random.Random) -> np.ndarray:
    if count <= 0:
        return arr
    out = arr.copy()
    h, w = out.shape
    for _ in range(count):
        y = rng.randint(0, h - 1)
        thickness = rng.randint(1, 2)
        intensity = rng.choice([0, 30, 60])
        out[y : min(h, y + thickness), :] = np.minimum(
            out[y : min(h, y + thickness), :], intensity
        )
    return out


def _jpeg_round_trip(arr: np.ndarray, quality: int) -> np.ndarray:
    quality = max(1, min(100, quality))
    ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return arr
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def degrade_image(image: Image.Image, profile: DegradationProfile) -> Image.Image:
    """Run the full degradation pipeline against a single page image."""

    rng = random.Random(profile.seed)
    arr = _to_cv(image)

    if profile.contrast != 1.0 or profile.brightness != 0:
        arr = arr.astype(np.float32) * profile.contrast + profile.brightness
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    arr = _rotate(arr, profile.rotation_deg, rng)

    if profile.blur_kernel and profile.blur_kernel >= 3:
        k = profile.blur_kernel | 1  # force odd
        arr = cv2.GaussianBlur(arr, (k, k), 0)

    if profile.gaussian_noise_sigma > 0:
        seed = rng.randrange(2**31)
        np_rng = np.random.default_rng(seed)
        noise = np_rng.normal(0, profile.gaussian_noise_sigma, arr.shape)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    arr = _add_speckle(arr, profile.speckle_density, rng)
    arr = _add_streaks(arr, profile.horizontal_streaks, rng)

    if profile.jpeg_quality < 95:
        arr = _jpeg_round_trip(arr, profile.jpeg_quality)

    return _to_pil(arr)
