import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

# EXTENSIONS COMPATIBLES
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


# BASE64 <-> PIL
def b64_to_pil(b64_str: str) -> Image.Image:
    """Convertit une image base64 en PIL.Image"""
    return Image.open(BytesIO(base64.b64decode(b64_str))).convert("RGB")


def pil_to_b64(img_pil: Image.Image, ext: str = ".png") -> str:
    """
    Convertit une image PIL en base64.
    L’extension détermine le format d’encodage.
    """
    if ext.lower() not in IMAGE_EXTENSIONS:
        ext = ".png"  # valeur par défaut

    buffer = BytesIO()
    img_pil.save(buffer, format=ext.replace(".", "").upper())
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# BASE64 <-> OpenCV
def b64_to_cv(b64_str: str) -> np.ndarray:
    """Convertit une image base64 en image OpenCV (BGR)"""
    img_bytes = base64.b64decode(b64_str)
    img_array = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def cv_to_b64(img_cv: np.ndarray, ext: str = ".png") -> str:
    """
    Convertit une image OpenCV en base64.
    L’extension détermine le format d’encodage.
    """
    if ext.lower() not in IMAGE_EXTENSIONS:
        ext = ".png"

    _, buffer = cv2.imencode(ext, img_cv)
    return base64.b64encode(buffer).decode("utf-8")


# PIL <-> OpenCV
def pil_to_cv(img_pil: Image.Image) -> np.ndarray:
    """Convertit PIL → OpenCV (RGB → BGR)"""
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def cv_to_pil(img_cv: np.ndarray) -> Image.Image:
    """Convertit OpenCV → PIL (BGR → RGB)"""
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


# Conversions couleur
def cv_to_gray(img_cv: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)


def cv_to_rgb(img_cv: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)


def cv_to_bgr(img_cv: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)


# Base64 UTF-8 safety
def ensure_utf8_base64(b64_value):
    """Garantit que la valeur base64 est bien une chaîne UTF-8"""
    if isinstance(b64_value, bytes):
        return b64_value.decode("utf-8")
    return str(b64_value)


def b64_to_html_img(b64_string, width=200):
    return f'<img src="data:image/png;base64,{b64_string}" width="{width}px">'