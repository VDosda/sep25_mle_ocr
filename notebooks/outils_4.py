import os

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

CLASSES = ["email", "form", "invoice", "presentation", "resume", "scientific_report"]

#cache
ARTEFACTS_30K = "../data/artifacts_rvl_cdip_filtered_dataset_2"
ARTEFACTS_60K = "../data/artifacts_rvl_cdip_filtered_dataset_3"
ARTEFACTS_DATASET_1 = "../data/artifacts_document-classification-dataset_1"


def list_images(dossier):
    lignes = []
    for racine, _, fichiers in os.walk(dossier):
        for fichier in fichiers:
            if fichier.endswith(".png"):
                lignes.append({"filepath": os.path.join(racine, fichier),
                               "label": os.path.basename(racine)})
    return pd.DataFrame(lignes)


def classify_by_cosine(X_train, y_train, X_test):
    """centroïde de chaque classe = moyenne de ses vecteurs du train ; un document prend la classe du centroïde
    le plus proche (cosinus). Renvoie les prédictions et le tableau des similarités."""
    if hasattr(X_train, "toarray"):  # le TF-IDF arrive en matrice creuse
        X_train = X_train.toarray()
    y_train = np.asarray(y_train)
    classes = sorted(set(y_train))
    centroides = normalize(np.vstack([X_train[y_train == classe].mean(axis=0) for classe in classes]))
    similarites = pd.DataFrame(cosine_similarity(X_test, centroides), columns=classes)
    return similarites.idxmax(axis=1).values, similarites


def load_ocr_sample():
    df = pd.read_csv(f"{ARTEFACTS_30K}/ocr_echantillon_1800.csv", keep_default_na=False)
    return train_test_split(df, test_size=0.2, random_state=1, stratify=df["label"])


def load_corpus_60k():
    df = pd.read_csv(f"{ARTEFACTS_60K}/ocr_corpus.csv", keep_default_na=False)
    df = df[df["text"] != ""].reset_index(drop=True)
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def top_two(scores, classes):
    ordre = np.argsort(-scores, axis=1)
    lignes = np.arange(len(scores))
    return pd.DataFrame({
        "top1": np.array(classes)[ordre[:, 0]],
        "top2": np.array(classes)[ordre[:, 1]],
        "ecart": scores[lignes, ordre[:, 0]] - scores[lignes, ordre[:, 1]],
    })


def load_image_features(filepaths, dossier_artefacts=ARTEFACTS_60K):
    pixels = pd.read_parquet(f"{dossier_artefacts}/image_features.parquet")
    mise_en_page = pd.read_parquet(f"{dossier_artefacts}/layout_features.parquet").drop(columns="label")
    return pixels.merge(mise_en_page, on="doc_id").set_index("doc_id").loc[list(filepaths)]


def preprocess_for_ocr(chemin, largeur_min=1500):
    """l'image comme l'OCR la reçoit (src/features/ocr.py) : noir et blanc, agrandie si elle est trop petite"""
    image = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hauteur, largeur = image.shape
    if largeur < largeur_min:
        echelle = largeur_min / largeur
        image = cv2.resize(image, (int(largeur * echelle), int(hauteur * echelle)), interpolation=cv2.INTER_CUBIC)
    return image


def table_features(chemin):
    """4 mesures de tableau : part de pixels sur des lignes horizontales / verticales, et nombre de lignes"""
    _, encre = cv2.threshold(preprocess_for_ocr(chemin), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hauteur, largeur = encre.shape
    # on ne garde que les longs traits continus
    lignes_h = cv2.morphologyEx(encre, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, largeur // 30), 1)))
    lignes_v = cv2.morphologyEx(encre, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, hauteur // 30))))
    n_h, _ = cv2.connectedComponents((lignes_h > 0).astype(np.uint8))
    n_v, _ = cv2.connectedComponents((lignes_v > 0).astype(np.uint8))
    return {"ratio_lignes_h": (lignes_h > 0).mean(), "ratio_lignes_v": (lignes_v > 0).mean(),
            "n_lignes_h": n_h - 1, "n_lignes_v": n_v - 1}


def save_probas(fichier, filepaths, probas):
    df = pd.DataFrame(probas, columns=CLASSES)
    df.insert(0, "filepath", list(filepaths))
    df.to_parquet(fichier, index=False)


def load_probas(fichier, filepaths):
    return pd.read_parquet(fichier).set_index("filepath").loc[list(filepaths), CLASSES].to_numpy()
