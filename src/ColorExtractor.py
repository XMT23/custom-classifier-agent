from dataclasses import dataclass
from typing import List, Union

import numpy as np

from src.utils_data import read_extended_dataset, crop_images, visualize_k_means

from .Kmeans import KMeans, get_colors


@dataclass
class ColorExtractorOptions:
    km_init: str = "first"
    verbose: bool = False
    tolerance: float = 0.00001
    max_iter: int = 300
    max_K: int = 5
    fitting: str = "WCD"
    best_k_tolerance: float = 0.2
    use_cropped_images: bool = True
    skin_filter: bool = True

    def __post_init__(self):
        if not (0 < self.best_k_tolerance <= 1):
            raise ValueError(
                f"best_k_tolerance must be between 0 and 1 (got {self.best_k_tolerance})"
            )


class ColorExtractor:
    def __init__(self, options: ColorExtractorOptions | None) -> None:
        self._options: ColorExtractorOptions = options or ColorExtractorOptions()

    def _crop_images(self) -> np.ndarray:
        imgs, _, _, upper, lower, _ = read_extended_dataset(
            root_folder="./data/raw/images",
            extended_gt_json="./data/raw/images/gt_reduced.json",
        )

        cropped_images = crop_images(imgs, upper, lower)
        return cropped_images

    def _rgb_to_hsv(self, rgb_img: np.ndarray) -> np.ndarray:
        if rgb_img.max() > 1.0:
            rgb_img = rgb_img / 255.0

        R, G, B = rgb_img[..., 0], rgb_img[..., 1], rgb_img[..., 2]
        max_c = np.max(rgb_img, axis=-1)
        min_c = np.min(rgb_img, axis=-1)
        delta = max_c - min_c

        # Hue
        H = np.zeros_like(max_c)
        idx = (max_c == R) & (delta != 0)
        H[idx] = 60 * ((G[idx] - B[idx]) / delta[idx]) % 360
        idx = (max_c == G) & (delta != 0)
        H[idx] = 60 * ((B[idx] - R[idx]) / delta[idx]) + 120
        idx = (max_c == B) & (delta != 0)
        H[idx] = 60 * ((R[idx] - G[idx]) / delta[idx]) + 240

        # saturation
        S = np.zeros_like(max_c)
        S[max_c != 0] = delta[max_c != 0] / max_c[max_c != 0]

        # Value
        V = max_c

        return np.stack([H, S, V], axis=-1)

    def _extract_foreground_pixels(self, img: np.ndarray) -> np.ndarray:
        hsv = self._rgb_to_hsv(img)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # Arbritary values!
        is_skin = (H >= 0) & (H <= 25) & (S >= 0.25) & (S <= 0.55) & (V >= 0.35)

        # Filter white background
        is_background = (V > 0.92) & (S < 0.08)

        valid_mask = ~(is_skin | is_background)

        pixels = img[valid_mask]

        if pixels.shape[0] == 0:
            pixels = img.reshape(-1, 3)

        return pixels

    def extract_dominant_colors(
        self, data: np.ndarray | None = None
    ) -> Union[List[str], List[List[str]]]:
        arr = None
        if self._options.use_cropped_images:
            print(
                "ALERT! 'use_cropped_images' option is set to true, "
                "all the database will be used"
            )
            arr = self._crop_images()
        else:
            arr = np.array(data, dtype=float)

        is_individual = arr.ndim == 3
        if is_individual:
            arr = np.expand_dims(arr, axis=0)

        all_dominant_colors = []
        opts = self._options
        for img in arr:
            pixels = img
            if self._options.skin_filter:
                pixels = self._extract_foreground_pixels(img)

            kmeans = KMeans(
                pixels,
                1,
                opts.km_init,
                opts.tolerance,
                opts.max_iter,
                opts.fitting,
                opts.best_k_tolerance,
            )

            kmeans.find_bestK(opts.max_K)
            dominant_labels = get_colors(kmeans.centroids)
            unique_colors = list(dict.fromkeys(dominant_labels))  # little hack
            all_dominant_colors.append(unique_colors)

        if is_individual:
            return all_dominant_colors[0]
        return all_dominant_colors
