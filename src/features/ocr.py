"""OCR (Tesseract) de tous les documents du projet. À lancer une fois, depuis la racine : python src/features/ocr.py

Les documents de documents_ocr.csv sont toujours lus dans le même ordre, sinon le découpage train / test change.
Compter 1 seconde par image ; ce qui est déjà fait n'est pas refait.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import pandas as pd
import pytesseract

DATASET_1 = "data/artifacts_document-classification-dataset_1"
EXTRAIT_1 = "data/artifacts_rvl_cdip_filtered_dataset_2"
EXTRAIT_FINAL = "data/artifacts_rvl_cdip_filtered_dataset_3"

# (corpus dans documents_ocr.csv, langue, fichier texte, fichier boîtes)
A_FAIRE = [
    ("dataset_1", "eng", f"{DATASET_1}/ocr_corpus.csv", f"{DATASET_1}/ocr_boxes.parquet"),
    ("dataset_2_extrait_1_partie_3", "eng", f"{EXTRAIT_1}/ocr_corpus.csv", f"{EXTRAIT_1}/ocr_boxes.parquet"),
    ("dataset_2_extrait_1_partie_4", "eng", f"{EXTRAIT_1}/ocr_echantillon_1800.csv", f"{EXTRAIT_1}/ocr_boxes_1800.parquet"),
    ("dataset_2_extrait_final", "eng", f"{EXTRAIT_FINAL}/ocr_corpus.csv", f"{EXTRAIT_FINAL}/ocr_boxes.parquet"),
    # notre ancien réglage (français) sur les mêmes 1 800 documents, pour mesurer le bug de langue (4.2)
    ("dataset_2_extrait_1_partie_4", "fra", f"{EXTRAIT_1}/ocr_echantillon_1800_fra.csv", None),
]


def preprocess_for_ocr(chemin, largeur_min=1500):
    image = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # noir et blanc (Otsu)
    hauteur, largeur = image.shape
    if largeur < largeur_min:  # Tesseract lit mal les petites images, on agrandit
        echelle = largeur_min / largeur
        image = cv2.resize(image, (int(largeur * echelle), int(hauteur * echelle)), interpolation=cv2.INTER_CUBIC)
    return image


def read_words(document):
    """une ligne par mot lu : sa boîte, sa confiance, son texte"""
    chemin, classe, langue = document
    image = preprocess_for_ocr(chemin)
    lu = pd.DataFrame(pytesseract.image_to_data(image, lang=langue, config="--psm 6", output_type=pytesseract.Output.DICT))
    mots = lu.loc[lu["level"] == 5, ["block_num", "par_num", "line_num", "word_num", "left", "top", "width", "height", "conf", "text"]]
    if mots.empty:  # rien de lu : on garde quand même une ligne pour ne pas perdre le document
        mots = pd.DataFrame([{**dict.fromkeys(mots.columns[:8], 0), "conf": -1, "text": ""}])
    mots = mots.astype({"conf": float, "text": str})
    mots.insert(0, "doc_id", chemin)
    mots.insert(1, "label", classe)
    return mots.assign(img_w=image.shape[1], img_h=image.shape[0])


def words_to_text(boites, confiance_min=60):
    """texte d'un document = ses mots lus avec assez de confiance, mis bout à bout"""
    mots = boites.assign(text=boites["text"].str.strip())
    mots = mots[(mots["text"] != "") & (mots["conf"] >= confiance_min)]
    return mots.groupby("doc_id")["text"].agg(" ".join)


if __name__ == "__main__":
    documents = pd.read_csv("src/features/documents_ocr.csv")
    for corpus, langue, fichier_texte, fichier_boites in A_FAIRE:
        if all(os.path.exists(f) for f in [fichier_texte, fichier_boites] if f):
            print(f"{corpus} ({langue}) : déjà fait")
            continue
        lot = documents[documents["corpus"] == corpus]
        print(f"{corpus} ({langue}) : OCR de {len(lot)} documents...")
        with ProcessPoolExecutor() as executeur:  # un document par cœur
            boites = pd.concat(executeur.map(read_words, zip(lot["filepath"], lot["label"], [langue] * len(lot)), chunksize=20),
                               ignore_index=True)

        os.makedirs(os.path.dirname(fichier_texte), exist_ok=True)
        if fichier_boites:
            boites.to_parquet(fichier_boites, index=False)
        texte = lot[["filepath", "label"]].copy()
        texte["text"] = texte["filepath"].map(words_to_text(boites)).fillna("")
        texte.to_csv(fichier_texte, index=False)
