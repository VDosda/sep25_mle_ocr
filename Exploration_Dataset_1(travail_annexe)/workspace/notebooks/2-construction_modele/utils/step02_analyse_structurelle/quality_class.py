import cv2
import numpy as np


class SegmentQualityEvaluator:
    """Qualité d'un segment : netteté, contraste, luminosité, ratio et score global."""

    def sharpness(self, roi):
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            return 0.0

    def contrast(self, roi):
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            return float(np.std(gray))
        except Exception:
            return 0.0

    def brightness(self, roi):
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            return float(np.mean(gray))
        except Exception:
            return 0.0

    def size(self, roi):
        try:
            h, w = roi.shape[:2]
            return w * h
        except Exception:
            return 0

    def ratio(self, roi):
        try:
            h, w = roi.shape[:2]
            return w / h if h > 0 else 0.0
        except Exception:
            return 0.0

    def global_score_old(self, roi):
        s = self.sharpness(roi)
        c = self.contrast(roi)
        b = self.brightness(roi)

        return float(
            (s / 1000) * 0.5 +
            (c / 50) * 0.3 +
            (b / 255) * 0.2
        )
    
    def global_score(self, roi):
        # Sharpness
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception:
            sharp = 0.0

        # Contrast
        try:
            contrast = float(np.std(gray))
        except Exception:
            contrast = 0.0

        # Brightness
        try:
            brightness = float(np.mean(gray))
        except Exception:
            brightness = 0.0

        # Entropy (texture richness)
        try:
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            hist_norm = hist / (hist.sum() + 1e-8)
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-8))
        except Exception:
            entropy = 0.0

        # Normalisation min-max
        sharp_n = min(sharp / 500.0, 1.0)
        contrast_n = min(contrast / 50.0, 1.0)
        brightness_n = brightness / 255.0
        entropy_n = entropy / 8.0  # entropy max ~8 pour 8 bits

        # Score géométrique (équilibré)
        score = (sharp_n * contrast_n * entropy_n) ** (1/3)

        return float(score)


    def evaluate(self, roi):
        return {
            "sharpness": self.sharpness(roi),
            "contrast": self.contrast(roi),
            "brightness": self.brightness(roi),
            "size": self.size(roi),
            "ratio": self.ratio(roi),
            "quality_score": self.global_score(roi)
        }

    def evaluate_dataframe(self, df_segments):
        results = []

        for _, row in df_segments.iterrows():
            metrics = self.evaluate(row["segment"])
            results.append({
                "_idx": row["_idx"],
                "segment_index": row["segment_index"],
                "quality": metrics
            })

        return results
