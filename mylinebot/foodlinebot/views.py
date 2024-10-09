import cv2
from scipy.spatial.distance import euclidean
from scipy.spatial import procrustes
from scipy.optimize import linear_sum_assignment
import random
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import os
import csv
from collections import deque
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
import imutils
import dlib
import ast
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseForbidden
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from linebot import LineBotApi, WebhookParser
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextSendMessage, ImageMessage, TextMessage

line_bot_api = LineBotApi(settings.LINE_CHANNEL_ACCESS_TOKEN)
parser = WebhookParser(settings.LINE_CHANNEL_SECRET)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, 'human_images')
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


def load_emo_data(csv_file_path):
    emo_dict = {}
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            if row:
                emo_dict[row[0]] = row[1]
    return emo_dict


emo_dict = load_emo_data('foodlinebot\emo.csv')


@csrf_exempt
def callback(request):
    if request.method != 'POST':
        return HttpResponseBadRequest("Only POST requests are allowed.")

    signature = request.META.get('HTTP_X_LINE_SIGNATURE', '')
    body = request.body.decode('utf-8')

    try:
        events = parser.parse(body, signature)
    except InvalidSignatureError:
        return HttpResponseForbidden("Invalid signature provided. Please check your LINE channel settings.")
    except LineBotApiError as e:
        return HttpResponseBadRequest(f"LINE Bot API error: {e}")

    for event in events:
        if isinstance(event, MessageEvent):
            if isinstance(event.message, ImageMessage):
                handle_image_message(event)
            elif isinstance(event.message, TextMessage):
                handle_text_message(event)
            else:
                handle_default_message(event)
        else:
            handle_default_message(event)

    return HttpResponse("OK")


def handle_image_message(event):
    try:
        image_content = line_bot_api.get_message_content(event.message.id)
        image_path = os.path.join(IMAGE_DIR, f'{event.message.id}.jpg')
        with open(image_path, 'wb') as file:
            for chunk in image_content.iter_content():
                file.write(chunk)

        processed_image_url = get_path_return_output(image_path)
        emo_text = emo_dict.get(processed_image_url, "No matching text found.")
        print(f"Processed image URL: {processed_image_url}")
        print(f"Emo text: {emo_text}")

        messages = [
            TextSendMessage(text=emo_text),
            TextSendMessage(text="請在1到10之間給這張圖片評分。")
        ]
        line_bot_api.reply_message(event.reply_token, messages)
    except Exception as e:
        error_message = "請傳送可清楚辨識的人臉照片"
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text=error_message))


def handle_text_message(event):
    user_id = event.source.user_id
    try:
        rating = int(event.message.text.strip())
        if 1 <= rating <= 10:
            append_to_csv(user_id, rating)
            thank_you_message = "感謝您的評分！😊"
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text=thank_you_message))
        else:
            prompt_message = "請输入1到10之間的數字來評分。"
            line_bot_api.reply_message(
                event.reply_token, TextSendMessage(text=prompt_message))
    except ValueError:
        prompt_message = "請輸入數字來評分。"
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text=prompt_message))


def append_to_csv(user_id, rating):
    with open('foodlinebot/rating.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([user_id, rating])


def handle_default_message(event):
    message = "目前只支持圖片消息和數字評分。"
    line_bot_api.reply_message(
        event.reply_token, TextSendMessage(text=message))


detector = dlib.get_frontal_face_detector()
shape_predictor = r'foodlinebot\shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(shape_predictor)
weights = {
    "euclidean": 1/2,
    "min_max_avg": 0,  # 永遠0
    "density": 0,
    "knn": 0,
    "Kuhn-Munkres": 0,  # 永遠0
    "EMD": 1/2,
    "jaccard": 0,
    "procrustes": 0,
}

max_values = {
    "euclidean": np.sqrt((400 - 150) ** 2 + (300 - 50) ** 2),
    "min_max_avg": np.sqrt((400 - 150) ** 2 + (300 - 50) ** 2),
    "density": 68,
    "knn": 1700,

}


def load_data():
    pointsdata = pd.read_csv("foodlinebot\points.csv")
    return pd.DataFrame(pointsdata)


df = load_data()


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
    points = np.array(points)
    x_min, x_max = x_range
    y_min, y_max = y_range
    points[:, 0] = (points[:, 0] - points[:, 0].min()) / \
        (points[:, 0].max() - points[:, 0].min()) * (x_max - x_min) + x_min
    points[:, 1] = (points[:, 1] - points[:, 1].min()) / \
        (points[:, 1].max() - points[:, 1].min()) * (y_max - y_min) + y_min

    return points


def calculate_euclidean_distance(shapes1, shapes2):
    tree2 = cKDTree(shapes2)
    distances, _ = tree2.query(shapes1, k=1)
    return np.mean(distances)


def calculate_similarity(shapes1, shapes2):
    shapes1 = np.array(shapes1)
    shapes2 = np.array(shapes2)

    cost_matrix = np.linalg.norm(shapes1[:, np.newaxis] - shapes2, axis=2)

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
    shapes1 = np.array(shapes1)
    shapes2 = np.array(shapes2)
    dist_matrix = np.linalg.norm(shapes1[:, np.newaxis] - shapes2, axis=2)
    density_list = np.sum(dist_matrix < radius, axis=1)
    return density_list.tolist()


def calculate_knn_distance(shapes1, shapes2, k=1):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(shapes2)

    distances, _ = neigh.kneighbors(shapes1)

    return np.mean(distances)


def EMD(shapes1, shapes2):
    shapes1 = np.array(shapes1)
    shapes2 = np.array(shapes2)

    cost_matrix = np.linalg.norm(shapes1[:, np.newaxis] - shapes2, axis=2)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_distance = cost_matrix[row_ind, col_ind].sum()
    max_distance = np.max(cost_matrix) * len(shapes1)
    similarity = 100 - (total_distance / max_distance * 100)
    return similarity


def calculate_jaccard_similarity(shapes1, shapes2, threshold):
    shapes1 = np.array(shapes1)
    shapes2 = np.array(shapes2)

    dist_matrix = np.linalg.norm(shapes1[:, np.newaxis] - shapes2, axis=2)

    similar_pairs = np.sum(dist_matrix < threshold)
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
    density = calculate_density(shapes1, shapes2)
    knn_distance = calculate_knn_distance(shapes1, shapes2, k=50)
    emd = EMD(shapes1, shapes2)
    jaccard = calculate_jaccard_similarity(shapes1, shapes2, threshold=42.5)
    procrustes = calculate_procrustes_similarity(shapes1, shapes2)

    euclidean_score = max(0, min(
        100, (max_values['euclidean'] - euclidean_distance) / max_values['euclidean'] * 100))
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
    for num in range(0, 150):
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
    folder_path = 'foodlinebot\images'
    scores = process_folder(shapes1, weights, max_values)
    highest_score_image, max_score = max(scores.items(), key=lambda x: x[1])
    print(
        f"The image with the highest score: {highest_score_image} {max_score}")
    highest_score_image_path = os.path.join(folder_path, highest_score_image)
    print(f"Returning path: {highest_score_image_path}")
    return highest_score_image_path
