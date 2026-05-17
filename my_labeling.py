__authors__ = ["1748951", "1755033", "1703660"]
__group__ = "11"

# from quant_analysis import kmeans_stats, get_shape_accuracy, get_color_accuracy
# from Kmeans import KMeanOptions, KMeans, get_colors
# from KNN import *
import numpy as np

from src import *

if __name__ == "__main__":
    (
        train_imgs,
        train_class_labels,
        train_color_labels,
        test_imgs,
        test_class_labels,
        test_color_labels,
    ) = read_dataset(
        root_folder="./data/raw/images/", gt_json="./data/raw/images/gt.json"
    )

    opts = ColorExtractorOptions(
        km_init="first",
        fitting="WCD",
        tolerance=0.001,
        best_k_tolerance=0.3,
        use_cropped_images=True,
    )
    color_extractor = ColorExtractor(opts)
    colors_database = color_extractor.extract_dominant_colors()
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset(
        root_folder="./data/raw/images", extended_gt_json="./data/raw/images/gt_reduced.json"
            )

    for i, colors in enumerate(colors_database):
        print(f"Colors extracted: {colors} - ground truth: {test_color_labels[i]}")

    accuracy = get_color_accuracy(colors_database, color_labels)

    exit()

if __name__ == "__main2__":
    # Load all the images and GT
    (
        train_imgs,
        train_class_labels,
        train_color_labels,
        test_imgs,
        test_class_labels,
        test_color_labels,
    ) = read_dataset(
        root_folder="./data/raw/images/", gt_json="./data/raw/images/gt.json"
    )

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # Convertim a gris per KNN
    train_imgs_gray = np.mean(train_imgs, axis=3).astype(float)
    test_imgs_gray = np.mean(test_imgs, axis=3).astype(float)

    # You can start coding your functions here
    # kmeans_stats(train_imgs, 8)

    # knn = KNN(train_imgs_gray, train_class_labels)
    # predicted_shapes = knn.predict(test_imgs_gray, k=5)
    # get_shape_accuracy(predicted_shapes, test_class_labels)

    predicted_colors = []
    for img in cropped_images:
        pixels = img.reshape(-1, 3).astype(float)
        kmeans_options = KMeanOptions(
            km_init="random", tolerance=0.001, fitting="ELBOW"
        )
        km = KMeans(pixels, K=1, options=kmeans_options)
        km.find_bestK(5)
        km.fit()
        predicted_colors.append(list(set(get_colors(km.centroids))))
    # Notese que en usar las cropped, la accuract subio de un 32 a un 59!
    get_color_accuracy(predicted_colors, color_labels)

    exit()

## FUNCIONS D'ANÀLISI QUALITATIU

## FUNCIÓ 1: RETRIEVAL_BY_COLOR (busquem peces de roba del mateix color)


def Retrieval_by_color(llista_imatges, etiquetes, query, percentatges=False):
    resultat = []
    percent = []

    if isinstance(query, str):
        query = [query]

    for img, labels in zip(llista_imatges, etiquetes):
        if all(color in labels for color in query):
            resultat.append(img)

            if percentatges:
                total_percent = sum(
                    np.sum(np.array(labels) == color) / len(labels) for color in query
                )
                percent.append(total_percent)

    if percentatges and resultat:
        sorted_results = sorted(
            zip(percent, resultat), key=lambda x: x[0], reverse=True
        )
        percent, resultat = map(list, zip(*sorted_results))

    resultat = np.array(resultat)

    if len(resultat) > 0:
        visualize_retrieval(resultat, topN=8, title=f"Retrieval by color: {query}")
    else:
        print(f"No s'han trobat imatges amb els colors: {query}")

    return resultat


# PROVA

# imatges_trobades = Retrieval_by_color(test_imgs, test_color_labels, ['Red','Blue'] , percentatges=True)

# FUNCIÓ 2: RETRIEVAL_BY SHAPE (Busquem peces de roba del mateix estil)


def Retrieval_by_shape(llista_imatges, etiquetes, query, neighbours=None, k=5):
    resultat = []
    percent = []

    for img, labels, veins in zip(
        llista_imatges,
        etiquetes,
        neighbours if neighbours is not None else [None] * len(etiquetes),
    ):
        if labels == query:
            resultat.append(img)

            if neighbours is not None:
                votes = np.sum(veins == query)
                percent.append(votes / k)

    if neighbours is not None and resultat:
        sorted_results = sorted(
            zip(percent, resultat), key=lambda x: x[0], reverse=True
        )
        percent, resultat = map(list, zip(*sorted_results))

    resultat = np.array(resultat)

    if len(resultat) > 0:
        visualize_retrieval(resultat, topN=8, title=f"Retrieval by shape: {query}")
    else:
        print(f"No s'han trobat imatges del tipus: {query}")

    return resultat

    # Passem les imatges en gris pel init del KNN. Si les passesim a color cada imatge ocuparia 60x80x3=14400 pixels en comptes de 60x80=4800

    peces_trobades = Retrieval_by_shape(
        test_imgs, predicted_shape_labels, "Shirts", neighbours=knn.neighbors, k=5
    )

    # FUNCIÓ 3:


def Retrieval_combined(
    llista_imatges, et_color, et_shape, color_query, shape_query, percentatges=False
):
    resultat = []
    percent = []

    if isinstance(color_query, str):
        color_query = [color_query]

    for img, et_color, et_shape in zip(llista_imatges, et_color, et_shape):
        # Comprovem que coincideix tant el color com la forma
        if all(color in et_color for color in color_query) and et_shape == shape_query:
            resultat.append(img)

            if percentatges:
                total_percent = sum(
                    np.sum(np.array(et_color) == color) / len(et_color)
                    for color in color_query
                )
                percent.append(total_percent)

    if percentatges and resultat:
        sorted_results = sorted(
            zip(percent, resultat), key=lambda x: x[0], reverse=True
        )
        percent, resultat = map(list, zip(*sorted_results))

    resultat = np.array(resultat)

    if len(resultat) > 0:
        visualize_retrieval(
            resultat, topN=8, title=f"Retrieval combined: {color_query} {shape_query}"
        )
    else:
        print(f"No s'han trobat imatges de {shape_query} amb colors: {color_query}")

    return resultat


Retrieval_combined(
    test_imgs,
    test_color_labels,
    predicted_shape_labels,
    "Black",
    "Sandals",
    percentatges=True,
)
