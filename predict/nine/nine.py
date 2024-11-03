import os

import numpy as np

from PIL import Image
from io import BytesIO
import onnxruntime as ort

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "resnet18.onnx")
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name


def read_icon():
    folder_path = os.path.join(current_dir, "icon")
    icon_list = []
    for i in range(91):
        file_path = os.path.join(folder_path, f"{i}.jpg")
        icon = Image.open(file_path)
        icon_list.append(np.array(icon))
    return icon_list


icon_list = read_icon()


def get_target_id(target_image):
    target_array = np.array(target_image)
    for i, icon_array in enumerate(icon_list):
        err = np.sum((target_array.astype("float") - icon_array.astype("float")) ** 2)
        err /= float(target_array.shape[0] * target_array.shape[1])
        if err < 1000:
            return i


def crop_image(image_bytes, coordinates):
    cropped_images = []
    img = Image.open(BytesIO(image_bytes))
    width, height = img.size

    left = 0
    upper = width
    right = height - width
    lower = height
    box = (left, upper, right, lower)
    icon_img = img.crop(box)

    grid_edge_length = width // 3
    for coord in coordinates:
        x, y = coord
        left = (x - 1) * grid_edge_length
        upper = (y - 1) * grid_edge_length
        right = left + grid_edge_length
        lower = upper + grid_edge_length
        box = (left, upper, right, lower)
        cropped_img = img.crop(box)
        cropped_images.append(cropped_img)
    return icon_img, cropped_images


def nine(image_bytes):
    coordinates = [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 1),
        (3, 2),
        (3, 3),
    ]

    def data_transforms(image):
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = image_array.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std
        image_array = np.transpose(image_array, (2, 0, 1))
        # image_array = np.expand_dims(image_array, axis=0)
        return image_array

    icon_image, cropped_image = crop_image(image_bytes, coordinates)
    target_id = get_target_id(icon_image)
    target_images = [data_transforms(i) for i in cropped_image]
    outputs = session.run(None, {input_name: target_images})[0]
    class_ids = np.argmax(outputs, axis=1).tolist()

    ans = []
    for i, id in enumerate(class_ids):
        if id == target_id:
            ans.append(coordinates[i])
    return ans
