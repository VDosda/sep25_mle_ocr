import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from .filters import is_mirrored

def brightness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) / 255.0

def sharpness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def contrast_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.std(gray)


def brightness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) / 255.0


def noise_score(img):
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    diff = cv2.absdiff(img, denoised)
    return np.mean(diff)


def orientation_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    return np.mean(np.abs(sobel))

def is_mirrored(img):
    """Détecte une image en miroir, avec une fenêtre SSIM adaptée à la taille de l'image."""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Déterminer le win_size maximal possible
    min_side = min(h, w)

    # win_size doit être impair et <= min_side
    if min_side < 3:
        return False  # image trop petite pour SSIM

    # win_size dynamique
    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size -= 1  # doit être impair

    # Flip horizontal
    flipped = cv2.flip(gray, 1)

    try:
        score, _ = ssim(gray, flipped, full=True, win_size=win_size)
        return score > 0.90
    except Exception:
        return False


def mirror_score(img):
    return 1.0 if is_mirrored(img) else 0.0


def text_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
    dark_ratio = np.mean(thresh == 0)
    return dark_ratio


def uniformity_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.var(gray)


def document_ratio(img, cropped=None):
    """Part de l'image originale occupée par le document recadré (entre 0 et 1)."""

    # Dimensions de l'image originale
    h, w = img.shape[:2]
    area_img = h * w

    # Détection du crop
    if cropped is None or cropped.shape[:2] == img.shape[:2]:
        return np.nan   # pas croppée → ratio inconnu

    # Dimensions du crop
    hc, wc = cropped.shape[:2]
    area_crop = hc * wc

    # Ratio de surface
    ratio = area_crop / area_img

    # entre 0 et 1
    return max(0.0, min(1.0, ratio))
