__authors__ = ["1748951", "1755033", "1703660"]
__group__ = "11"

import numpy as np

from src import *
from src.quant_analysis import get_shape_accuracy

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

    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset(
        root_folder="./data/raw/images",
        extended_gt_json="./data/raw/images/gt_reduced.json",
    )

    cropped = crop_images(imgs, upper, lower)

    print("[1/3] Running Color Extractor in test set")
    opts_color = ColorExtractorOptions(
        km_init="custom",
        fitting="FISHER",
        tolerance=0.0,
        max_iter=1000,
        max_K=3,
        best_k_tolerance=0.2,
        use_cropped_images=False,
        skin_filter=True,
    )
    color_extractor = ColorExtractor(opts_color)
    predicted_color_labels = color_extractor.extract_dominant_colors(test_imgs)
    color_accuracy = get_color_accuracy(predicted_color_labels, test_color_labels)
    print(f"-> Accuracy color extractor: {color_accuracy}%")

    print("\n[2/3] Training Shape Labeler and classifying clothes")
    opts_shape = ShapeLabelerOptions(
        apply_gray_transform=True, distance_fn=MANHATTAN_METRIC
    )
    labeler = ShapeLabeler(train_imgs, train_class_labels, opts_shape)
    predicted_shape_labels = labeler.predict(test_imgs, k=3)
    knn_instance = labeler._knn
    shape_accuracy = get_shape_accuracy(predicted_shape_labels, test_class_labels)

    print("\n[3/3] Running query system")
    while True:
        print("\n" + "=" * 50)
        print("\t\tSearch Engine")
        print("=" * 50)
        print("1. Search by Color")
        print("2. Search by Cloth type (Shape)")
        print("3. Search combined")
        print("4. Exit")
        print("=" * 50)

        option = input("Select an option (1-4): ").strip()

        match option:
            case "1":
                print("\n--- Search by Color ---")
                print(
                    "Avalaible colors: Black, White, Red, Blue, Green, Yellow, Gray, Pink, Purple, Orange, Brown"
                )
                query_color = input(
                    "Enter desired color/s (for multiple, separate by commas): "
                ).strip()

                if query_color:
                    color_list = [
                        c.strip().capitalize() for c in query_color.split(",")
                    ]
                    print(f"\nSearching for clothes with color/s: {color_list}...")
                    _ = Retrieval_by_color(
                        test_imgs, predicted_color_labels, color_list, percentatges=True
                    )

            case "2":
                print("\n--- Search by Shape ---")
                print(
                    "Avalaible shapes: Dresses, Flip Flops, Jeans, Sandals, Shirts, Shorts, Socks, Handbags"
                )
                query_shape = input("Enter desired shape: ").strip().title()

                if query_shape:
                    print(f"\nSearching for clothes with shape: {query_shape}...")
                    _ = Retrieval_by_shape(
                        test_imgs,
                        predicted_shape_labels,
                        query_shape,
                        neighbours=knn_instance.neighbors,
                        k=3,
                    )

            case "3":
                print("\n--- Combined Search ---")
                query_color = input(
                    "Enter desired color/s (for multiple, separate by commas): "
                ).strip()
                query_shape = input("Enter desired shape: ").strip().title()

                if query_color and query_shape:
                    color_list = [
                        c.strip().capitalize() for c in query_color.split(",")
                    ]
                    _ = Retrieval_combined(
                        test_imgs,
                        predicted_color_labels,
                        predicted_shape_labels,
                        color_list,
                        query_shape,
                        percentatges=True,
                    )
            case "4":
                print("\nThank you for using the clothes search engine! Exiting...")
                break
            case _:
                print("Invalid option. Please, enter a number from 1 to 4")
