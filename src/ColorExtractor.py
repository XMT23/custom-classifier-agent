from dataclasses import dataclass
from typing import List, Union

import numpy as np

from src.utils_data import read_extended_dataset, crop_images

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

    def __post_init__(self):
        if not (0 < self.best_k_tolerance <= 1):
            raise ValueError(
                f"best_k_tolerance must be between 0 and 1 (got {self.best_k_tolerance})"
            )


class ColorExtractor:
    def __init__(self, options: ColorExtractorOptions | None) -> None:
        self._options: ColorExtractorOptions = options or ColorExtractorOptions()

    def _crop_images(self) -> np.ndarray:
        imgs, _, __, upper, lower, background = read_extended_dataset(
            root_folder="./data/raw/images",
            extended_gt_json="./data/raw/images/gt_reduced.json",
        )

        cropped_images = crop_images(imgs, upper, lower)
        return cropped_images

    def extract_dominant_colors(
        self, data: np.ndarray | None = None
    ) -> Union[List[str], List[List[str]]]:
        arr = None
        if self._options.use_cropped_images:
            print("ALERT! 'use_cropped_images' option is set to true, "
                  "all the database will be used")
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
            if opts.use_cropped_images:
                pixels = img.reshape(-1, 3).astype(float)
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
            all_dominant_colors.append(list(dominant_labels))

        if is_individual:
            return all_dominant_colors[0]
        return all_dominant_colors
