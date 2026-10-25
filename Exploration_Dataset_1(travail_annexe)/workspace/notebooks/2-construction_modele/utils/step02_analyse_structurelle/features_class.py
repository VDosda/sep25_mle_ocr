import cv2
import numpy as np


class FeatureExtractor:
    """Features de chaque segment : HOG et histogrammes."""

    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog_size = 3780  # taille standard HOG 128x64

    def _is_valid_roi(self, roi):
        if roi is None or not isinstance(roi, np.ndarray):
            return False
        if roi.size == 0:
            return False
        h, w = roi.shape[:2]
        return h >= 5 and w >= 5

    def ratio(self, roi):
        h, w = roi.shape[:2]
        return w / h if h > 0 else 0.0

    def histogram(self, roi):
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            return cv2.normalize(hist, hist).flatten()
        except Exception:
            return np.zeros(32, dtype=np.float32)

    def contrast(self, roi):
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            return float(np.std(gray))
        except Exception:
            return 0.0

    def hog_features(self, roi):
        h, w = roi.shape[:2]

        if h < 64 or w < 128:
            return np.zeros(self.hog_size, dtype=np.float32)

        try:
            roi_resized = cv2.resize(roi, (128, 64))
            hog_vec = self.hog.compute(roi_resized)
            if hog_vec is None:
                return np.zeros(self.hog_size, dtype=np.float32)
            return hog_vec.flatten()
        except Exception:
            return np.zeros(self.hog_size, dtype=np.float32)

    def stats(self, roi):
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            return float(np.mean(gray)), float(np.var(gray))
        except Exception:
            return 0.0, 0.0

    def extract_features(self, roi):
        if not self._is_valid_roi(roi):
            return {
                "valid": False,
                "ratio": 0.0,
                "contrast": 0.0,
                "mean": 0.0,
                "var": 0.0,
                **{f"hist_{i}": 0.0 for i in range(32)},
                **{f"hog_{i}": 0.0 for i in range(self.hog_size)}
            }

        features = {"valid": True}

        features["ratio"] = self.ratio(roi)
        features["contrast"] = self.contrast(roi)

        mean, var = self.stats(roi)
        features["mean"] = mean
        features["var"] = var

        hist = self.histogram(roi)
        for i, v in enumerate(hist):
            features[f"hist_{i}"] = float(v)

        hog_vec = self.hog_features(roi)
        for i, v in enumerate(hog_vec):
            features[f"hog_{i}"] = float(v)

        return features

    def extract_from_dataframe(self, df_segments):
        results = []

        for _, row in df_segments.iterrows():
            feats = self.extract_features(row["segment"])
            results.append({
                "_idx": row["_idx"],
                "segment_index": row["segment_index"],
                "features": feats
            })

        return results
