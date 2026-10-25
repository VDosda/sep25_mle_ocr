import cv2
import numpy as np

# 1. Resize universel (non destructif)
def cv_resize_max(img, max_dim=1600):
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)

    if scale >= 1.0:
        return img  # déjà assez petit

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# 2. Unrotate (portrait vs paysage)
def cv_unrotate(img):
    h, w = img.shape[:2]

    # Si portrait → OK
    if h > w:
        return img

    # Détection orientation via HoughLines
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    if lines is None:
        return img

    angles = [theta for rho, theta in lines[:,0]]
    mean_angle = np.mean(angles)

    # Si angle proche de vertical → rotate 90°
    if abs(mean_angle - np.pi/2) < 0.2:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    return img


# 2b. Détection miroir
def is_mirrored(img):
    """Détecte une image en miroir : corrélation entre l'image et sa version retournée."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flipped = cv2.flip(gray, 1)

    gray_norm = gray.astype(np.float32) / 255.0
    flipped_norm = flipped.astype(np.float32) / 255.0

    corr = np.mean(gray_norm * flipped_norm)

    # Seuil empirique : > 0.85 = image miroir probable
    return corr > 0.85


# 2c. Correction miroir
def fix_mirror(img):
    return cv2.flip(img, 1)


# 3. Denoise léger (non destructif)
def cv_denoise(img, strength=0.2):
    return cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=10 * strength,
        hColor=10 * strength,
        templateWindowSize=7,
        searchWindowSize=21
    )


# 4. Autocrop (si document détecté)
def cv_autocrop(img):
    """Recadre sur le document, ou renvoie l'image d'origine si le recadrage échoue."""

    # image vide ou None
    if img is None or img.size == 0:
        return img

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Flou léger pour stabiliser la détection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Détection des bords
    edges = cv2.Canny(blur, 50, 150)

    # Trouver les contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img  # aucun contour → fallback

    # Prendre le plus grand contour
    cnt = max(contours, key=cv2.contourArea)

    # Si contour trop petit → fallback
    if cv2.contourArea(cnt) < 500:
        return img

    # Bounding box
    x, y, w, h = cv2.boundingRect(cnt)

    # éviter un crop vide
    if w <= 0 or h <= 0:
        return img

    cropped = img[y:y+h, x:x+w]

    # éviter une image vide
    if cropped is None or cropped.size == 0:
        return img

    return cropped

    
# 5. clahe (contraste léger)
def cv_clahe(img, strength=0.5):
    if img is None or img.size == 0:
        return img  # on renvoie l’image telle quelle

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clip = 2.0 + strength * 4.0
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


