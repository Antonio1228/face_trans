import cv2
import ast
import dlib
import imutils
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from collections import deque
import os
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
import random
from scipy.optimize import linear_sum_assignment
from scipy.spatial import procrustes
from scipy.spatial.distance import euclidean

detector = dlib.get_frontal_face_detector()
shape_predictor = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(shape_predictor)
weights = {
    "euclidean": 1/8,
    "min_max_avg": 1/8,
    "density": 1/8,
    "knn": 1/8,
    "Kuhn-Munkres": 1/8,
    "EMD": 1/8,
    "jaccard": 1/8,
    "procrustes": 1/8,
}

max_values = {
    "euclidean": np.sqrt((400 - 150) ** 2 + (300 - 50) ** 2),
    "min_max_avg": np.sqrt((400 - 150) ** 2 + (300 - 50) ** 2),
    "density": 68,
    "knn": 1700,

}


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


def shape_to_np(shape, image_height, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        x = shape.part(i).x
        y = image_height - shape.part(i).y
        coords[i] = (x, y)
    return coords


def extract_facial_landmarks(image_path, shape_predictor):
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=500)
    image_height = image.shape[0]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    shapes = []
    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape, image_height)
        shapes.append(shape)
    return image, shapes


def transform_points(points, x_range, y_range):
    min_x, max_x = x_range
    min_y, max_y = y_range

    orig_min_x = min(points, key=lambda p: p[0])[0]
    orig_max_x = max(points, key=lambda p: p[0])[0]
    orig_min_y = min(points, key=lambda p: p[1])[1]
    orig_max_y = max(points, key=lambda p: p[1])[1]

    transformed_points = []
    for x, y in points:
        new_x = min_x + (x - orig_min_x) / (orig_max_x -
                                            orig_min_x) * (max_x - min_x)
        new_y = min_y + (y - orig_min_y) / (orig_max_y -
                                            orig_min_y) * (max_y - min_y)
        transformed_points.append((new_x, new_y))

    return transformed_points


def calculate_euclidean_distance(shapes1, shapes2):
    distances = []
    for point1 in shapes1:
        min_distance = min(np.sqrt(
            (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) for point2 in shapes2)
        distances.append(min_distance)
    return np.mean(distances)


def calculate_similarity(shapes1, shapes2):
    cost_matrix = np.zeros((len(shapes1), len(shapes2)))
    for i, coord1 in enumerate(shapes1):
        for j, coord2 in enumerate(shapes2):
            cost_matrix[i, j] = np.linalg.norm(
                np.array(coord1) - np.array(coord2))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    total_distance = cost_matrix[row_ind, col_ind].sum()
    max_distance = np.max(cost_matrix) * len(shapes1)
    similarity = 100 - (total_distance / max_distance * 100)

    return similarity


def calculate_min_max_avg_distance(shapes1, shapes2):
    all_distances = []
    for point1 in shapes1:
        distances = [np.sqrt((point1[0] - point2[0]) ** 2 +
                             (point1[1] - point2[1]) ** 2) for point2 in shapes2]
        all_distances.append(min(distances))

    return np.min(all_distances), np.max(all_distances), np.mean(all_distances)


def calculate_density(shapes1, shapes2, radius=250):
    density_list = []
    for point1 in shapes1:
        count = sum(np.sqrt((point1[0] - point2[0]) ** 2 +
                    (point1[1] - point2[1]) ** 2) < radius for point2 in shapes2)
        density_list.append(count)
    return density_list


def calculate_knn_distance(shapes1, shapes2, k=1):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(shapes2)

    distances, _ = neigh.kneighbors(shapes1)

    return np.mean(distances)


def EMD(shapes1, shapes2):
    cost_matrix = np.zeros((len(shapes1), len(shapes2)))
    for i, coord1 in enumerate(shapes1):
        for j, coord2 in enumerate(shapes2):
            cost_matrix[i, j] = np.linalg.norm(
                np.array(coord1) - np.array(coord2))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_distance = cost_matrix[row_ind, col_ind].sum()
    max_distance = np.max(cost_matrix) * len(shapes1)
    similarity = 100 - (total_distance / max_distance * 100)

    return similarity


def calculate_jaccard_similarity(shapes1, shapes2, threshold):
    similar_pairs = 0
    for point1 in shapes1:
        if any(np.linalg.norm(np.array(point1) - np.array(point2)) < threshold for point2 in shapes2):
            similar_pairs += 1
    for point2 in shapes2:
        if any(np.linalg.norm(np.array(point2) - np.array(point1)) < threshold for point1 in shapes1):
            similar_pairs += 1
    similar_pairs = similar_pairs / 2
    union_size = len(shapes1) + len(shapes2) - similar_pairs
    jaccard_similarity = (similar_pairs / union_size) * 100

    return jaccard_similarity


def calculate_procrustes_similarity(shapes1, shapes2):
    mtx1 = np.array(shapes1)
    mtx2 = np.array(shapes2)
    mtx1_transformed, mtx2_transformed, disparity = procrustes(mtx1, mtx2)
    similarity_score = 1 / (1 + disparity)
    similarity_score_normalized = similarity_score * 100
    similarity_score_normalized = max(min(similarity_score_normalized, 100), 0)

    return similarity_score_normalized


def perform_comparisons(shapes1, shapes2, weights, max_values):
    euclidean_distance = calculate_euclidean_distance(shapes1, shapes2)
    min_distance, max_distance, avg_distance = calculate_min_max_avg_distance(
        shapes1, shapes2)
    density = calculate_density(shapes1, shapes2)
    knn_distance = calculate_knn_distance(shapes1, shapes2, k=50)
    KuhnMunkres = calculate_similarity(shapes1, shapes2)
    emd = EMD(shapes1, shapes2)
    jaccard = calculate_jaccard_similarity(shapes1, shapes2, threshold=42.5)
    procrustes = calculate_procrustes_similarity(shapes1, shapes2)

    euclidean_score = max(0, min(
        100, (max_values['euclidean'] - euclidean_distance) / max_values['euclidean'] * 100))
    min_max_avg_score = max(0, min(
        100, (max_values['min_max_avg'] - avg_distance) / max_values['min_max_avg'] * 100))
    density_score = max(
        0, min(100, np.mean(density) / max_values['density'] * 100))
    knn_score = max(
        0, min(100, (max_values['knn'] - knn_distance) / max_values['knn'] * 100))

    weighted_average_score = (
        weights["euclidean"] * euclidean_score +
        weights["density"] * density_score +
        weights["knn"] * knn_score +
        weights["EMD"] * emd +
        weights["jaccard"] * jaccard +
        weights["procrustes"] * procrustes
    )

    return weighted_average_score


def process_folder(shapes1, weights, max_values):
    # get scores of all picture
    scores = {}
    pointsdata = pd.read_csv("points.csv")
    df = pd.DataFrame(pointsdata)
    for num in range(0, 110):
        shapes2 = df.iloc[num].to_list()
        image_file = shapes2[0]
        shapes2.remove(image_file)
        shapes2 = [ast.literal_eval(point) for point in shapes2]
        x_range = (150, 400)
        y_range = (50, 300)
        shapes1_new = transform_points(shapes1[0], x_range, y_range)
        shapes2_new = transform_points(shapes2, x_range, y_range)
        score = perform_comparisons(
            shapes1_new, shapes2_new, weights, max_values)
        scores[image_file] = score
        print(f"Score for {image_file}: {score:.2f}")
    return scores


def get_path_return_output(image_path1):
    image1, shapes1 = extract_facial_landmarks(image_path1, shape_predictor)
    folder_path = 'images'
    scores = process_folder(shapes1, weights, max_values)
    highest_score_image, max_score = max(scores.items(), key=lambda x: x[1])
    print(
        f"The image with the highest score: {highest_score_image} {max_score}")
    highest_score_image_path = os.path.join(folder_path, highest_score_image)
    return highest_score_image, highest_score_image_path
