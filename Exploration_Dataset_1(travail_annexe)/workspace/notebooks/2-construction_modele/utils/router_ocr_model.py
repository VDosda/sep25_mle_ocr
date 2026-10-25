def choose_ocr_model_features_only(features: dict) -> str:
    """
    Choix automatique du meilleur modèle OCR selon les features.
    Retourne un nom de modèle : 'paddle', 'tesseract', 'easyocr', 'opencv', 'transformer'
    """

    sharpness = features.get("sharpness", 0)
    contrast = features.get("contrast", 0)
    text_density = features.get("text_density", 0)
    edge_density = features.get("edge_density", 0)
    brightness = features.get("brightness", 128)
    crop_ratio = features.get("crop_ratio", 1.0)
    orientation = features.get("orientation_angle", 0)

    # 1. Cas très dégradé → Transformers (TrOCR / Donut)
    if sharpness < 40 and contrast < 20:
        return "transformer"

    # 2. Documents nets, bien contrastés → PaddleOCR
    if sharpness > 120 and contrast > 40:
        return "paddle"

    # 3. Documents très textuels → Tesseract
    if text_density > 0.25 and sharpness > 80:
        return "tesseract"

    # 4. Documents faibles mais lisibles → EasyOCR
    if sharpness > 60 and contrast > 25:
        return "easyocr"

    # 5. Cas très simple → OpenCV (contours / digits)
    if edge_density > 0.15 and text_density < 0.10:
        return "opencv"

    # 6. Default fallback
    return "easyocr"

def choose_ocr_model(features, qualite_score):
    q = qualite_score
    sharp = features["sharpness"]
    bright = features["brightness"]
    contrast = features["contrast"]
    density = features["text_density"]

    # 1. Images très propres → Paddle
    if q > 70 and sharp > 500 and 40 < bright < 240:
        return "paddle"

    # 2. Images moyennes → EasyOCR
    if 55 < q <= 70:
        return "easyocr"

    # 3. Images très textuelles → Tesseract
    if density > 0.35 and sharp > 400:
        return "tesseract"

    # 4. Images dégradées → Transformer OCR
    if q < 55 or sharp < 300 or contrast < 20:
        return "transformer"

    # 5. Images très simples → OpenCV OCR
    if density < 0.15 and contrast > 30:
        return "opencv"

    # Fallback
    return "easyocr"


# Définition des zones attendues par modèle
zones = {
    "paddle": (70, 100),
    "easyocr": (55, 70),
    "opencv": (0, 55)
}

def routing_coherence(row):
    model = row["best_model"]
    q = row["quality_score"]

    if model not in zones:
        return np.nan  # modèle non évalué

    low, high = zones[model]
    return 1 if low <= q <= high else 0


