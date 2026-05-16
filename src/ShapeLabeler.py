from typing import Union
import numpy as np
from .KNN import KNN, DistanceMetric, EUCLIDEAN_METRIC
from dataclasses import dataclass

@dataclass
class ShapeLabelerOptions:
    reduce_train_set: bool = False
    apply_gray_transform: bool = False
    reduction_scale: float = 0.5  # Scale for 'reduced' feature mode
    distance_fn: DistanceMetric = EUCLIDEAN_METRIC


class ShapeLabeler:
    def __init__(
        self,
        train_imgs: np.ndarray,
        train_labels: list | np.ndarray,
        options: ShapeLabelerOptions | None,
    ) -> None:
        """
        train_imgs must be a 4-dimensiolan matrix (P x H x W x 3):
            P = number of imgs
            H = height
            W = width
            3 = 3 RGB channels
        """
        if train_imgs.ndim != 4:
            raise ValueError(
                "Expected a 4-dimensional array.\n"
                "'train_imgs must be "
                "a 4 dimensional array of shape (PxHxWxC) "
                "where: \n\tP = number of imgs"
                "\n\tH = height"
                "\n\tW = width"
                "\n\tC = RGB channels"
                f"\n\nGot dimensions {train_imgs.ndim}"
            )
        if train_imgs.shape[-1] != 3:
            raise ValueError("Final axis must be 3 channels (RGB)")

        self._train_labels: np.ndarray = np.array(train_labels)
        self._train_images: np.ndarray = np.array(train_imgs, dtype=float)
        self._train_images_gray: np.ndarray = self._normalize_gray_scale(train_imgs)
        self._options: ShapeLabelerOptions = options or ShapeLabelerOptions()
        self._knn = self._instanciate_knn()

    @property
    def options(self) -> ShapeLabelerOptions:
        return self._options

    @options.setter
    def options(self, new_options: ShapeLabelerOptions) -> None:
        self._options = new_options
        self._knn = self._instanciate_knn()

    def _instanciate_knn(self) -> KNN:
        opts: ShapeLabelerOptions = self.options

        train_labels = self._train_labels
        train_imgs = self._train_images
        if opts.apply_gray_transform:
            train_imgs = self._train_images_gray

        knn = KNN(train_imgs, train_labels, opts.distance_fn)
        return knn

    def _normalize_gray_scale(self, images: np.ndarray) -> np.ndarray:
        arr = np.array(images, dtype=float)

        is_individual = arr.ndim == 3
        if is_individual:
            # Consider the iamge as a array of images of len 1
            arr = np.expand_dims(arr, axis=0)

        rgb_mean = np.mean(arr, axis=-1)
        arr_gray_3_channels = np.stack([rgb_mean, rgb_mean, rgb_mean], axis=-1)

        return arr_gray_3_channels

    def predict(self, data: np.ndarray, k: int = 5) -> Union[str, np.ndarray]:
        arr = np.array(data, dtype=float)

        is_individual = arr.ndim == 3
        if is_individual:
            arr = np.expand_dims(arr, axis=0)

        if self.options.apply_gray_transform:
            arr = self._normalize_gray_scale(arr)

        predictions = self._knn.predict(arr, k=k)

        if is_individual:
            return predictions[0]
        return predictions
