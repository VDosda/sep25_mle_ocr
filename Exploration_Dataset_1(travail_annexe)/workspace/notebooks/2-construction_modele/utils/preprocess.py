import cv2
import numpy as np

from .filters import (
    cv_resize_max,
    cv_denoise,
    cv_sharpen,
    cv_threshold,
    cv_unrotate,
    cv_deskew,
    cv_autocrop,
    cv_clahe,
    cv_super_resolution
)

def preprocess(img_cv):
    # 1. Resize universel (non destructif)
    img = cv_resize_max(img_cv, 1600)

    # 2. Unrotate (portrait vs paysage)
    img = cv_unrotate(img)

    # 3. Denoise léger (réduit le bruit sans lisser le texte)
    img = cv_denoise(img, strength=0.2)

    # 4. Autocrop (si document détecté)
    img = cv_autocrop(img)

    # 5. CLAHE léger (améliore le contraste local sans créer de bruit)
    img = cv_clahe(img, strength=0.1)

    # 6. Extraire les features universelles (densité, segments, etc.)
    features = extract_features(img)

    return img, features


def preprocess_paddle(img):
    img = cv_resize_max(img, 960)
    img = cv_denoise(img)
    img = cv_sharpen(img)
    return img

def preprocess_tesseract(img):
    img = cv_resize_max(img, 1024)
    img = cv_denoise(img)
    img = cv_threshold(img, method="otsu")
    return img

def preprocess_google(img):
    img = cv_resize_max(img, 1600)
    img = cv_denoise(img, strength=0.2)
    return img

def preprocess_opencv(img):
    img = cv_resize_max(img, 960)
    img = cv_sharpen(img)
    img = cv_threshold(img, method="adaptive")
    return img

PREPROCESS_SWITCH = {
    "none": preprocess,
    "paddle": preprocess_paddle,
    "tesseract": preprocess_tesseract,
    "google": preprocess_google,
    "opencv": preprocess_opencv
}

import cv2
import numpy as np

def extract_features(img):
    """
    Analyse l'image préprocessée pour extraire les features utiles
    au routeur OCR (CPU-only, EAST comme détecteur principal).
    """

    features = {}

    # 1. Résolution
    h, w = img.shape[:2]
    features["width"] = w
    features["height"] = h
    features["max_dim"] = max(h, w)
    features["min_dim"] = min(h, w)

    # 2. Contraste global (écart-type de l'histogramme)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features["contrast_std"] = float(np.std(gray))

    # 3. Niveau de bruit (variance du Laplacien)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    features["noise_laplacian_var"] = float(np.var(lap))

    # 4. Orientation (détection de rotation globale)
    # On mesure l'orientation dominante via Sobel + PCA
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag = np.sqrt(sobelx**2 + sobely**2)
    coords = np.column_stack(np.where(mag > np.percentile(mag, 95)))

    if len(coords) > 50:
        _, eigenvectors = cv2.PCACompute(coords.astype(np.float32), mean=None)
        angle = np.degrees(np.arctan2(eigenvectors[0,1], eigenvectors[0,0]))
    else:
        angle = 0.0

    features["orientation_angle"] = float(angle)

    # 5. Détection des segments via EAST (détecteur principal)
    net = cv2.dnn.readNet("models/east.pb")

    blob = cv2.dnn.blobFromImage(
        img, 1.0, (320, 320),
        (123.68, 116.78, 103.94),
        swapRB=True, crop=False
    )
    net.setInput(blob)

    scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid",
                                    "feature_fusion/Conv_7/Conv_7"])

    # Décodage des boîtes EAST
    rects = []
    confidences = []
    height, width = scores.shape[2:4]

    for y in range(height):
        for x in range(width):
            score = scores[0, 0, y, x]
            if score < 0.5:
                continue

            offsetX, offsetY = x * 4.0, y * 4.0
            angle = geometry[0, 4, y, x]
            cosA = np.cos(angle)
            sinA = np.sin(angle)

            h = geometry[0, 0, y, x]
            w = geometry[0, 1, y, x]

            endX = int(offsetX + (cosA * w + sinA * h))
            endY = int(offsetY + (sinA * w - cosA * h))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(float(score))

    # NMS
    indices = cv2.dnn.NMSBoxes(rects, confidences, 0.5, 0.4)
    segments = [rects[i[0]] for i in indices]

    # 6. Features sur les segments
    features["segments_count"] = len(segments)

    if len(segments) > 0:
        widths = [abs(x2 - x1) for (x1, y1, x2, y2) in segments]
        heights = [abs(y2 - y1) for (x1, y1, x2, y2) in segments]

        features["segment_width_mean"] = float(np.mean(widths))
        features["segment_height_mean"] = float(np.mean(heights))
        features["segment_width_std"] = float(np.std(widths))
        features["segment_height_std"] = float(np.std(heights))

        # Dispersion = distance moyenne entre les centres
        centers = [( (x1+x2)/2, (y1+y2)/2 ) for (x1,y1,x2,y2) in segments]
        if len(centers) > 1:
            dists = []
            for i in range(len(centers)-1):
                for j in range(i+1, len(centers)):
                    dists.append(np.linalg.norm(np.array(centers[i]) - np.array(centers[j])))
            features["segments_dispersion"] = float(np.mean(dists))
        else:
            features["segments_dispersion"] = 0.0
    else:
        features["segment_width_mean"] = 0.0
        features["segment_height_mean"] = 0.0
        features["segment_width_std"] = 0.0
        features["segment_height_std"] = 0.0
        features["segments_dispersion"] = 0.0

    return features


def score_preprocessing_quality(features):
    """
    Calcule un score global de qualité du preprocessing entre 0 et 100.
    Les features sont celles retournées par extract_features().
    """

    # Normalisations (min/max empiriques)
    sharpness_norm = min(features.get("sharpness", 0) / 150.0, 1.0)
    contrast_norm = min(features.get("contrast", 0) / 60.0, 1.0)

    brightness = features.get("brightness", 128)
    brightness_norm = 1.0 - abs(brightness - 128) / 128.0
    brightness_norm = max(min(brightness_norm, 1.0), 0.0)

    edge_density_norm = min(features.get("edge_density", 0) * 10, 1.0)
    text_density_norm = min(features.get("text_density", 0) * 3, 1.0)

    crop_ratio_norm = min(features.get("crop_ratio", 0) * 1.5, 1.0)

    orientation_angle = features.get("orientation_angle", 0)
    orientation_norm = 1.0 if orientation_angle == 0 else 0.7

    # pondérations
    weights = {
        "sharpness": 0.20,
        "contrast": 0.15,
        "brightness": 0.10,
        "edge_density": 0.10,
        "text_density": 0.15,
        "crop_ratio": 0.15,
        "orientation": 0.15
    }

    score = (
        sharpness_norm * weights["sharpness"] +
        contrast_norm * weights["contrast"] +
        brightness_norm * weights["brightness"] +
        edge_density_norm * weights["edge_density"] +
        text_density_norm * weights["text_density"] +
        crop_ratio_norm * weights["crop_ratio"] +
        orientation_norm * weights["orientation"]
    )

    return round(score * 100, 2)
