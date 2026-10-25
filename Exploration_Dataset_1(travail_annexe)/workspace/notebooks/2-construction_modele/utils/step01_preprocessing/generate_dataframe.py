import time
from functools import partial
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.common import b64_to_cv

from utils.common.filters import (
    cv_resize_max,
    cv_unrotate,
    is_mirrored,
    fix_mirror,
    cv_denoise,
    cv_autocrop,
    cv_clahe
)
from utils.common.features import (
    sharpness_score,
    contrast_score,
    brightness_score,
    noise_score,
    orientation_score,
    mirror_score,
    text_density,
    uniformity_score,
    document_ratio,
)

def adjust_brightness(img, target=0.0, strength=0.5):
    """
    Ajuste la luminosité via gamma correction.
    target : brightness souhaité (0 = neutre)
    strength : intensité de la correction (0.5 = modéré)
    """
    current = brightness_score(img)
    delta = current - target

    # gamma < 1 = éclaircit, gamma > 1 = assombrit
    gamma = 1.0 + strength * delta

    gamma = max(0.3, min(gamma, 3.0))  # sécurité

    # correction gamma
    img_float = img.astype(np.float32) / 255.0
    img_corr = np.power(img_float, gamma)
    return (img_corr * 255).astype(np.uint8)

# PROFILING SIMPLE
def _now():
    return time.perf_counter()

# TRAITEMENT ROW

def _process_row(row: pd.Series, step: str) -> dict:
    """Traite une ligne (image_filename, image_base64) et renvoie un dict au format de preprocessing_schema.json."""
    
    idx = row["_idx"]  # on récupère l’index original
    
    img_cv = b64_to_cv(row["image_base64"])
    
    # Prétraitement léger
    img_cv = cv_resize_max(img_cv)
    img_cv = cv_unrotate(img_cv)

    if is_mirrored(img_cv):
        img_cv = fix_mirror(img_cv)

    img_cv = cv_denoise(img_cv)
    cropped = cv_autocrop(img_cv)
        
    img_cv = cv_clahe(cropped)
    
    img_cv =adjust_brightness(img_cv)

    # Features
    sharp = sharpness_score(img_cv)
    cont = contrast_score(img_cv)
    bright = brightness_score(img_cv)
    noise = noise_score(img_cv)
    orient = orientation_score(img_cv)
    mirror = mirror_score(img_cv)
    dens = text_density(img_cv)
    unif = uniformity_score(img_cv)
    doc_r = document_ratio(img_cv)
    
   
    
    # Extraction des features
    features = {
        "_idx": idx,
        "image_filename": row["image_filename"],
        "image_base64": row["image_base64"],
        "pipeline_step": step,

        "sharpness": sharp,
        "contrast": cont,
        "brightness": bright,
        "noise": noise,
        "orientation": orient,
        "mirror": mirror,
        "density": dens,
        "uniformity": unif,
        "doc_ratio": doc_r,
    }

    return features


# TRAITEMENT D’UN DATAFRAME COMPLET AVEC BARRE DE PROGRESSION
def generate_preprocessing_dataframe(df: pd.DataFrame, step, use_multiprocessing: bool = True):
    """Prétraite chaque image de df, puis calcule ses features et son score de qualité (multiprocessing en option)."""

    t0 = _now()

    records = df.to_dict("records")

    if use_multiprocessing:
        # Nombre de workers = nb de CPU logiques
        n_workers = max(1, cpu_count() - 1)

        process_fn = partial(_process_row, step=step)

        with Pool(processes=n_workers) as p:
            results = list(
                tqdm(
                    p.imap_unordered(process_fn, records),
                    total=len(records),
                    desc=f"Préprocessing images (mp, {n_workers} workers)",
                )
            )
    else:
        results = []
        for row_dict in tqdm(records, total=len(records), desc="Préprocessing images (single thread)"):
            results.append(_process_row(row_dict, step))

    t1 = _now()
    print(f"[generate_dataframe] Temps total : {t1 - t0:.2f} s pour {len(df)} images")

    return pd.DataFrame(results).sort_values("_idx").reset_index(drop=True)
