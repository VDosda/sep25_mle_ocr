import cv2
import numpy as np
from utils.common import b64_to_cv


class EastDetector:
    """Détecteur de zones de texte EAST (OpenCV)."""

    def __init__(self, model_path, width=320, height=320):
        self.model_path = model_path
        self.width = width
        self.height = height

        self.output_layers = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"
        ]

        self.net = cv2.dnn.readNet(self.model_path)

    def load_image(self, row):
        return b64_to_cv(row["image_base64"])

    def preprocess(self, img):
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (self.width, self.height)),
            scalefactor=1.0,
            size=(self.width, self.height),
            mean=(123.68, 116.78, 103.94),
            swapRB=True,
            crop=False
        )
        return blob

    def detect(self, img):
        blob = self.preprocess(img)
        self.net.setInput(blob)
        scores, geometry = self.net.forward(self.output_layers)

        num_rows, num_cols = scores.shape[2:4]
        boxes, confidences = [], []

        for y in range(num_rows):
            scores_data = scores[0, 0, y]
            x0_data = geometry[0, 0, y]
            x1_data = geometry[0, 1, y]
            x2_data = geometry[0, 2, y]
            x3_data = geometry[0, 3, y]
            angles_data = geometry[0, 4, y]

            for x in range(num_cols):
                score = scores_data[x]
                if score < 0.5:
                    continue

                angle = angles_data[x]
                cos, sin = np.cos(angle), np.sin(angle)

                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                offset_x, offset_y = x * 4.0, y * 4.0

                end_x = int(offset_x + cos * x1_data[x] + sin * x2_data[x])
                end_y = int(offset_y - sin * x1_data[x] + cos * x2_data[x])
                start_x, start_y = int(end_x - w), int(end_y - h)

                boxes.append((start_x, start_y, end_x, end_y))
                confidences.append(float(score))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        segments = []
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, x2, y2 = boxes[i]
                roi = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if roi.size > 0:
                    segments.append({
                        "bbox": (x1, y1, x2, y2),
                        "roi": roi
                    })

        return segments

    def extract_from_dataframe(self, df):
        results = []

        for _, row in df.iterrows():
            img = self.load_image(row)
            segments = self.detect(img)

            for i, seg in enumerate(segments):
                results.append({
                    "_idx": row["_idx"],
                    "segment_index": i,
                    "segment": seg["roi"],
                    "bbox": seg["bbox"]
                })

        return results
