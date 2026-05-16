__authors__ = ["1748951", "1755033", "1703660"]
__group__ = "11"

from dataclasses import dataclass
import sys
from typing import Optional
import numpy as np
import utils


@dataclass
class KMeanOptions:
    km_init: str = "first"
    verbose: bool = False
    tolerance: float = 0.0
    max_iter: int = sys.maxsize  # Big number aka no max iters
    fitting: str = "WCD"
    best_k_tolerance: float = 0.2

    def __post_init__(self):
        if not (0 < self.best_k_tolerance <= 1):
            raise ValueError(
                f"best_k_tolerance must be between 0 and 1 (got {self.best_k_tolerance})"
            )


class KMeans:
    def __init__(
        self, X: np.ndarray, K: int = 1, options: Optional[KMeanOptions] = None
    ):
        """
        Constructor of KMeans class
            Args:
                K (int): Number of cluster
                options (dict): dictionary with options
        """
        self.num_iter = 0
        self.K: int = K
        self._init_X(X)
        self.options: KMeanOptions = options or KMeanOptions()

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X: np.ndarray | list):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
        Args:
            X (list or np.array): list(matrix) of all pixel values
                if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        X = np.array(X, dtype=np.float64)

        if X.ndim == 3:
            # From F x C x 3 to N x 3
            F, C, triplet = X.shape
            X = X.reshape(F * C, triplet)

        self.X = X

    def _first_centroids(self) -> np.ndarray:
        """
        Take the first K distinct points from the image X and take them as centroids
        Returns:
            centroids (np.ndarray): array containing the first K centroids
        """

        centroids = []
        for pixel in self.X:
            if not any(np.array_equal(pixel, c) for c in centroids):
                centroids.append(pixel)
            if len(centroids) == self.K:
                break

        return np.array(centroids, np.float64)

    def _random_centroids(self) -> np.ndarray:
        """
        Take K distinct points randomly from image X and take them as centroids.
        Return:
            centroids (np.ndarray): array containing the K random centroids
        """

        centroids = []
        for i in np.random.permutation(len(self.X)):
            if not any(np.array_equal(self.X[i], c) for c in centroids):
                centroids.append(self.X[i])
            if len(centroids) == self.K:
                break
        return np.array(centroids, np.float64)

    def _custom_centroids(self) -> np.ndarray:
        # minims i maxims de cada dimensio (D,)
        min_vals = self.X.min(axis=0)
        max_vals = self.X.max(axis=0)

        # k punts equispaiats entre 0 i 1
        t = np.linspace(0, 1, self.K)  # [0, 1/(K-1), 2/(K-1), ..., 1]

        centroids = min_vals + t[:, None] * (max_vals - min_vals)  # (K, D)

        return np.array(centroids, dtype=np.float64)

    def _init_centroids(self):
        """
        Initialization of centroids
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        init_method = self.options.km_init
        match init_method.lower():
            case "first":
                self.centroids = self._first_centroids()
            case "random":
                self.centroids = self._random_centroids()
            case "custom":
                self.centroids = self._custom_centroids()
            case _:
                raise ValueError(f"Initilization option '{init_method}' invalid")

        self.old_centroids = np.zeros_like(self.centroids)

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        distances = distance(self.X, self.centroids)
        self.labels = np.argmin(distances, axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.old_centroids = np.copy(self.centroids)
        new_centroids = np.zeros((self.K, self.X.shape[1]))

        for i in range(self.K):
            pixels_in_cluster_i = self.X[self.labels == i]
            if pixels_in_cluster_i.shape[0] > 0:
                new_centroids[i] = np.mean(pixels_in_cluster_i, axis=0)
            else:
                new_centroids[i] = self.old_centroids[i]

        self.centroids = new_centroids

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        if self.num_iter >= self.options.max_iter:
            return True

        diff = np.abs(self.centroids - self.old_centroids)
        return np.all(diff <= self.options.tolerance)

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self._init_centroids()
        self.num_iter = 0

        while not self.converges():
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1

    def withinClassDistance(self):
        """
        returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pixel_centroids = self.centroids[self.labels]
        self.WCD = np.sum((self.X - pixel_centroids) ** 2) / self.X.shape[0]
        return self.WCD

    def interClassDistance(self):
        """
        Returns the inter-class distance: mean pairwise distance between centroids.
        We remember that higher ICD means clusters are more spread apart (better separation).
        """
        K = self.centroids.shape[0]
        if K < 2:
            self.ICD = 0.0
            return

        total = 0.0
        count = 0
        for i in range(K):
            for j in range(i + 1, K):
                total += np.sqrt(np.sum((self.centroids[i] - self.centroids[j]) ** 2))
                count += 1

        self.ICD = total / count

    def fisherCoefficient(self):
        """
        Returns Fisher's criterion: the ratio of between-class scatter to within-class scatter.
        Again, we remember that higher values indicate better-separated, compact clusters.
        """
        global_mean = np.mean(self.X, axis=0)

        # between-class scatter: weighted sum of squared distances from each centroid to global mean
        between = 0.0
        for i in range(self.K):
            n_i = np.sum(self.labels == i)
            diff = self.centroids[i] - global_mean
            between += n_i * np.dot(diff, diff)

        # within-class scatter: sum of squared distances from each point to its centroid
        within = 0.0
        for i in range(self.K):
            points = self.X[self.labels == i]
            if points.shape[0] > 0:
                diffs = points - self.centroids[i]
                within += np.sum(diffs**2)

        # NEcessary line, this way we can avoid a division by 0
        self.FISHER = between / within if within > 1e-10 else np.inf

    def find_bestK(self, max_K):
        """
        Sets the best K by analysing results up to 'max_K' clusters.
        Supports three fitting heuristics: 'WCD', 'ICD', 'FISHER'.
        """
        tolerance = self.options.best_k_tolerance
        fitting = self.options.fitting.upper()

        last_metric = None

        # We are gonna use the elbow method: computes WCD for all K values and finds the point of maximum curvature
        # using the second finite difference of the WCD curve. The elbow is where the rate of
        # decrease shifts most sharply, detected as argmax of delta^2(WCD).
        # we extracted the information from: https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
        if fitting == "ELBOW":
            wcds = []
            for K in range(2, max_K + 1):
                self.K = K
                self.fit()
                self.withinClassDistance()
                wcds.append(self.WCD)

            if len(wcds) < 3:
                self.K = max_K
                return

            # second finite difference: measures curvature at each interior point
            second_diff = [
                wcds[i - 1] - 2 * wcds[i] + wcds[i + 1] for i in range(1, len(wcds) - 1)
            ]
            best_idx = int(np.argmax(second_diff))
            self.K = (
                best_idx + 3
            )  # necesarry because K starts at 2, second_diff starts at index 1
            self.fit()
            return

        for K in range(2, max_K + 1):
            self.K = K
            self.fit()

            if fitting == "WCD":
                self.withinClassDistance()
                metric = self.WCD
                # if WCD decreases then we stop when relative improvement drops below tolerance
                if last_metric is not None:
                    if last_metric == 0 or (1 - metric / last_metric) < tolerance:
                        self.K = K - 1
                        return

            elif fitting == "ICD":
                self.interClassDistance()
                metric = self.ICD
                # On the other hand, if ICD increases then we stop when relative gain drops below tolerance
                if last_metric is not None and last_metric > 0:
                    if (metric / last_metric - 1) < tolerance:
                        self.K = K - 1
                        return

            elif fitting == "FISHER":
                self.fisherCoefficient()
                metric = self.FISHER
                # Finally, if Fisher increases we stop when relative gain drops below tolerance
                if last_metric is not None and last_metric > 0:
                    if (metric / last_metric - 1) < tolerance:
                        self.K = K - 1
                        return
            else:
                raise ValueError(
                    f"Fitting option '{fitting}' not recognised. Use 'WCD', 'ICD' or 'FISHER'."
                )

            last_metric = metric

        # we keep max_K if there is not an early stop


def distance(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set

    Nota:
        En un principi, el programa trigava molt a calcular la millor k i fer els fits.

        En revisar el codi i l'execució del programa, no es va trigar a veure que la major part
        del temps d'execució s'emprava en calcular les distàncies euclidianes.

        L'implementació original era aquesta:

        N = X.shape[0]
        K = C.shape[0]
        dist = np.zeros((N, K))
        for i in range(N):
            for j in range(K):
                # Eucl distance
                dist[i, j] = np.sqrt(np.sum((X[i] - C[j])**2))
        return dist

        Tractant de buscar una solució per aquest problema, ens hem topat amb aquesta
        conversa de StackOverflow:
        https://stackoverflow.com/questions/52366421/how-to-do-n-d-distance-and-nearest-neighbor-calculations-on-numpy-arrays?noredirect=1&lq=1

        La resposta proposa fer D = np.sqrt(np.sum((X[:, None, :] - Y[None, :, :])**2, axis = -1)).
        Tot i que indiqui que aquest mètode consumeix molta memòria, en el nostre cas ha reduït el temps
        d'execució del Test Suite d'uns 40 segons a aproximadament 0.7 segons.

        La idea és afegir una dimensió extra a les dues matrius sense canviar les dades internes de cadascuna:
            - X de (P, D) a (P, 1, D)
            - C de (K, D) a (1, K, D)

        Així, NumPy pot calcular les diferències directament sense necessitar un doble
        bucle de Python (horriblement lent) i s'aprofita de les operacions optimitzades
        de la llibreria implementada en C.
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################

    diff = X[:, None, :] - C[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def get_colors(centroids: np.ndarray):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    probs = utils.get_color_prob(centroids)
    idxs = np.argmax(probs, axis=1)
    return utils.colors[idxs]
