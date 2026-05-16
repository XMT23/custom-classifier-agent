import numpy as np
import matplotlib.pyplot as plt

import time
from Kmeans import KMeans


def kmeans_stats(imgs: np.ndarray, Kmax):
    wcds = []
    iters = []
    times = []

    n_samples = 100
    sample_idxs = np.random.choice(len(imgs), size=n_samples, replace=False)
    sample_imgs = imgs[sample_idxs]

    for K in range(2, Kmax + 1):
        k_wcds, k_iters, k_times = [], [], []

        for img in sample_imgs:
            pixels = img.reshape(-1, 3).astype(float)
            km = KMeans(pixels, K)

            start = time.time()
            km.fit()
            elapsed = time.time() - start

            k_wcds.append(km.withinClassDistance())
            k_iters.append(km.num_iter)
            k_times.append(elapsed)

        wcds.append(np.mean(k_wcds))
        iters.append(np.mean(k_iters))
        times.append(np.mean(k_times))

        print(
            "K="
            + str(K)
            + " | WCD="
            + str(round(wcds[-1], 4))
            + " | Iters="
            + str(round(iters[-1], 1))
            + " | Time="
            + str(round(times[-1], 4))
            + "s"
        )

    Ks = list(range(2, Kmax + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Kmeans Statistics", fontsize=14, fontweight="bold")

    axes[0].plot(Ks, wcds, marker="o", color="steelblue")
    axes[0].set_title("Within-Class Distance (WCD)")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("WCD")
    axes[0].grid(True)

    axes[1].plot(Ks, iters, marker="s", color="darkorange")
    axes[1].set_title("Iterations to Converge")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Iterations")
    axes[1].grid(True)

    axes[2].plot(Ks, times, marker="^", color="forestgreen")
    axes[2].set_title("Time to Converge (s)")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Time (s)")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    return {"K": Ks, "wcd": wcds, "iterations": iters, "times": times}


def get_shape_accuracy(predicted_labels, ground_truth_labels):
    predicted_labels = np.array(predicted_labels)
    ground_truth_labels = np.array(ground_truth_labels)

    if len(predicted_labels) != len(ground_truth_labels):
        raise ValueError(
            "predicted_labels and ground_truth_labels must have same length"
        )

    correct = np.sum(predicted_labels == ground_truth_labels)
    accuracy = (correct / len(ground_truth_labels)) * 100.0

    print(
        f"Shape Accuracy: {correct}/{len(ground_truth_labels)} correct → {accuracy:.2f}%"
    )
    return accuracy


def get_color_accuracy(predicted_colors, ground_truth_colors):
    if len(predicted_colors) != len(ground_truth_colors):
        raise ValueError(
            "predicted_colors and ground_truth_colors must have the same length."
        )

    jaccard_scores = []

    for pred, gt in zip(predicted_colors, ground_truth_colors):
        pred_set = set(pred)
        gt_set = set(gt)

        intersection = len(pred_set & gt_set)
        union = len(pred_set | gt_set)

        if union == 0:
            jaccard_scores.append(1.0)  # both empty → perfect match
        else:
            jaccard_scores.append(intersection / union)

    accuracy = np.mean(jaccard_scores) * 100.0

    print(f"Color Accuracy (mean Jaccard): {accuracy:.2f}%")
    return accuracy


# def kmeans_statistics(images: np.ndarray, Kmax: int) -> dict:
#     results = {
#         "k": [],
#         "wcd": [],
#         "time": [],
#         "iters": [],
#     }
#
#     for k in range(2, Kmax + 1):
#         kmeans_obj.K = k
#
#         start_time = time()
#         kmeans_obj.fit()
#         end_time = time()
#
#         results["k"].append(k)
#         results["wcd"].append(kmeans_obj.withinClassDistance())
#         results["iters"].append(kmeans_obj.num_iter)
#         results["time"].append(end_time - start_time)
#
#     plot_stats(results)
#     return results
#
# def plot_stats(results):
#     plt.figure(figsize=(12, 4))
#
#     plt.subplot(1, 3, 1)
#     plt.plot(results["k"], results["wcd"], marker="o")
#     plt.title("WCD vs K (Elbow methdod)")
#     plt.xlabel("K")
#
#     plt.subplot(1, 3, 2)
#     plt.plot(results["k"], results["time"], marker="^", color="orange")
#     plt.title("Time vs K")
#
#     plt.subplot(1, 3, 3)
#     plt.plot(results["k"], results["iters"], marker="*", color="green")
#     plt.title("Iterations vs K")
#
#     plt.tight_layout()
#     plt.show()
#
# def get_shape_accuracy(results, ground_truth) -> float:
#     pass
