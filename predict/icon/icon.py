import os
import ddddocr
from PIL import Image
from io import BytesIO
import numpy as np
import onnxruntime as ort

det = ddddocr.DdddOcr(det=True, ocr=False)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "siamese.onnx")
Siamese = ort.InferenceSession(model_path)


def get_icons(image_bytes):
    def remove_subicon(data):
        result = []
        for i in data:
            remove = False
            for j in data:
                if (
                    i != j
                    and i[0] >= j[0]
                    and i[1] >= j[1]
                    and i[2] <= j[2]
                    and i[3] <= j[3]
                ):
                    remove = True
            if not remove:
                result.append(i)
        return result

    img = Image.open(BytesIO(image_bytes))
    bboxes = det.detection(image_bytes)
    small_bboxes = [i for i in bboxes if i[1] > 300]
    big_bboxes = [i for i in bboxes if i[1] <= 300]
    small_bboxes = sorted(small_bboxes, key=lambda x: x[0])
    big_bboxes = remove_subicon(big_bboxes)
    small_images = [img.crop(i) for i in small_bboxes]
    big_images = [img.crop(i) for i in big_bboxes]
    return big_bboxes, small_images, big_images


def calculate_similarity(img1, img2):
    def preprocess_image(img, size=(105, 105)):
        img_resized = img.resize(size)
        img_normalized = np.array(img_resized) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_expanded = np.expand_dims(img_transposed, axis=0).astype(np.float32)
        return img_expanded

    image_data_1 = preprocess_image(img1)
    image_data_2 = preprocess_image(img2)

    inputs = {"input": image_data_1, "input.53": image_data_2}

    output = Siamese.run(None, inputs)

    output_sigmoid = 1 / (1 + np.exp(-output[0]))
    similarity_score = output_sigmoid[0][0]

    return similarity_score


def icon(image_bytes):
    big_bboxes, small_images, big_images = get_icons(image_bytes)
    ans = []
    for i in small_images:
        similarities = [calculate_similarity(i, j) for j in big_images]
        target_bbox = big_bboxes[similarities.index(max(similarities))]
        x = (target_bbox[0] + target_bbox[2]) / 2
        y = (target_bbox[1] + target_bbox[3]) / 2
        ans.append((x, y))
    return ans
