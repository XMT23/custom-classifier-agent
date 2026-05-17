__authors__ = ["1748951", "1755033", "1703660"]
__group__ = "11"

import numpy as np
from scipy.spatial.distance import cdist

from typing import Callable

DistanceMetric = Callable[[np.ndarray, np.ndarray], np.ndarray]

EUCLIDEAN_METRIC: DistanceMetric = lambda x, y: cdist(x, y, "euclidean")
MANHATTAN_METRIC: DistanceMetric = lambda x, y: cdist(x, y, "cityblock")
COSINE_METRIC: DistanceMetric = lambda x, y: cdist(x, y, "cosine")


class KNN:
    def __init__(
        self, train_data: np.ndarray, labels: np.ndarray, distance_fn: DistanceMetric
    ):
        self._init_train(train_data)
        self.distance_fn = distance_fn
        self.labels = np.array(labels)

    def _init_train(self, train_data: np.ndarray):
        train_data = np.array(train_data, dtype=float)
        self.original_shape = train_data.shape[1:3]  # (m, n)
        self.train_data = self._extract_features(train_data)

    def _extract_features(self, data: np.ndarray) -> np.ndarray:
        data = np.array(data, dtype=float)
        return data.reshape(data.shape[0], -1)

    def get_k_neighbours(self, test_data: np.ndarray, k: int) -> np.ndarray:
        test_data = np.array(test_data, dtype=float)
        test_feats = self._extract_features(test_data)

        distances = self.distance_fn(test_feats, self.train_data)

        sorted_idxs = np.argsort(distances, axis=-1)
        self.neighbors = self.labels[sorted_idxs[:, :k]]
        return self.neighbors

    def get_class(self) -> np.ndarray:
        predictions = []

        for row in self.neighbors:
            row_list = list(row)
            best_val = None
            max_votes = -1

            already_counted = []
            for label in row_list:
                if label not in already_counted:
                    votes = row_list.count(label)
                    if votes > max_votes:
                        max_votes = votes
                        best_val = label
                    already_counted.append(label)

            predictions.append(best_val)

        return np.array(predictions)

    def predict(self, test_data: np.ndarray, k: int) -> np.ndarray:
        self.get_k_neighbours(test_data, k)
        return self.get_class()
