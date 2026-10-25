"""Prépare RVL-CDIP-N, le jeu de test de la démo. À lancer une fois, depuis la racine : python src/features/rvl_cdip_n.py

RVL-CDIP-N (https://huggingface.co/datasets/stefan-hf/rvlcdip-n) : des documents récents du web, rangés dans les
classes de RVL-CDIP, que ni nos modèles ni le DiT n'ont vus. On garde email, form, invoice, resume et
scientific_publication : 383 documents. Compter quelques minutes ; ce qui est déjà fait n'est pas refait.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import pandas as pd
from huggingface_hub import snapshot_download
from PIL import Image

from ocr import read_words, words_to_text

CLASSES = ["email", "form", "invoice", "resume", "scientific_publication"]
DOSSIER_IMAGES = "data/rvl_cdip_n"
ARTEFACTS = "data/artifacts_rvl_cdip_n"


def pixel_features(chemin):
    """70 mesures sur les pixels, sans OCR (notebook 4.1_modele_image)"""
    gris = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)
    hauteur, largeur = gris.shape
    # otsu : encre en blanc, fond en noir
    _, encre = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # on ne garde que les traits de plus de 40 px (lignes de tableau)
    lignes_h = cv2.morphologyEx(encre, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))
    lignes_v = cv2.morphologyEx(encre, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40)))
    mesures = {"mean_intensity": gris.mean() / 255, "std_intensity": gris.std() / 255, "aspect_ratio": largeur / hauteur,
               "ink_ratio": (encre > 0).mean(), "h_line_ratio": (lignes_h > 0).mean(), "v_line_ratio": (lignes_v > 0).mean()}
    # grille 8x8 : quantité d'encre dans chaque case
    bords_y = [i * (hauteur // 8) for i in range(8)] + [hauteur]
    bords_x = [j * (largeur // 8) for j in range(8)] + [largeur]
    for i in range(8):
        for j in range(8):
            mesures[f"grid_{i}_{j}"] = (encre[bords_y[i]:bords_y[i + 1], bords_x[j]:bords_x[j + 1]] > 0).mean()
    return mesures


def layout_features(boites, documents):
    """15 mesures de mise en page, calculées sur les boîtes des mots de l'OCR (notebook 4.1_modele_image)"""
    mots = boites[(boites["width"] > 0) & (boites["height"] > 0)]
    mots = mots.assign(droite=mots["left"] + mots["width"], bas=mots["top"] + mots["height"])
    g = mots.groupby("doc_id")
    largeur, hauteur = g["img_w"].first(), g["img_h"].first()
    n_lignes = mots.drop_duplicates(["doc_id", "block_num", "par_num", "line_num"]).groupby("doc_id").size()
    grand = mots["height"] > 1.5 * g["height"].transform("median")  # mots bien plus hauts que les autres = titres
    mesures = pd.DataFrame({
        "n_words": g.size(), "n_blocks": g["block_num"].nunique(), "n_lines": n_lignes, "words_per_line": g.size() / n_lignes,
        "conf_mean": g["conf"].mean(), "conf_std": g["conf"].std(),
        "height_mean": g["height"].mean(), "height_std": g["height"].std(), "big_text_ratio": grand.groupby(mots["doc_id"]).mean(),
        "text_width_ratio": (g["droite"].max() - g["left"].min()) / largeur,
        "text_height_ratio": (g["bas"].max() - g["top"].min()) / hauteur,
        "top_margin": g["top"].min() / hauteur, "bottom_margin": (hauteur - g["bas"].max()) / hauteur,
        "left_dispersion": g["left"].std() / largeur})
    mesures = mesures.reindex(documents).fillna(0)  # aucun mot lu : tout à 0
    mesures["empty_ocr"] = (mesures["n_words"] == 0).astype(int)
    return mesures.astype({"n_words": int, "n_blocks": int, "n_lines": int})


if __name__ == "__main__":
    # les images : un dossier par classe, comme nos autres datasets
    if not os.path.exists(DOSSIER_IMAGES):
        print("Téléchargement de RVL-CDIP-N...")
        telechargement = snapshot_download("stefan-hf/rvlcdip-n", repo_type="dataset",
                                           allow_patterns=[f"test/{classe}/*.jpg" for classe in CLASSES])
        for classe in CLASSES:
            os.makedirs(f"{DOSSIER_IMAGES}/{classe}")
            for fichier in sorted(os.listdir(f"{telechargement}/test/{classe}")):
                image = Image.open(f"{telechargement}/test/{classe}/{fichier}").convert("L")  # noir et blanc, comme RVL-CDIP
                image.save(f"{DOSSIER_IMAGES}/{classe}/{fichier.replace('.jpg', '.png')}")

    documents = pd.DataFrame([{"filepath": f"{DOSSIER_IMAGES}/{classe}/{fichier}", "label": classe}
                              for classe in CLASSES for fichier in sorted(os.listdir(f"{DOSSIER_IMAGES}/{classe}"))])
    os.makedirs(ARTEFACTS, exist_ok=True)

    # OCR : un document par cœur
    if not os.path.exists(f"{ARTEFACTS}/ocr_corpus.csv"):
        print(f"OCR de {len(documents)} documents...")
        with ProcessPoolExecutor() as executeur:
            boites = pd.concat(executeur.map(read_words, zip(documents["filepath"], documents["label"], ["eng"] * len(documents)),
                                             chunksize=20), ignore_index=True)
        boites.to_parquet(f"{ARTEFACTS}/ocr_boxes.parquet", index=False)
        corpus = documents.copy()
        corpus["text"] = corpus["filepath"].map(words_to_text(boites)).fillna("")
        corpus.to_csv(f"{ARTEFACTS}/ocr_corpus.csv", index=False)

    # les 85 mesures : 70 sur les pixels, 15 de mise en page
    if not os.path.exists(f"{ARTEFACTS}/layout_features.parquet"):
        print("Les 85 mesures...")
        pixels = pd.DataFrame([pixel_features(f) for f in documents["filepath"]])
        pixels.insert(0, "doc_id", documents["filepath"])
        pixels.to_parquet(f"{ARTEFACTS}/image_features.parquet", index=False)

        mise_en_page = layout_features(pd.read_parquet(f"{ARTEFACTS}/ocr_boxes.parquet"), documents["filepath"])
        mise_en_page = mise_en_page.reset_index(names="doc_id")
        mise_en_page.insert(1, "label", documents["label"])
        mise_en_page.to_parquet(f"{ARTEFACTS}/layout_features.parquet", index=False)
    print(f"Prêt : {DOSSIER_IMAGES}/ et {ARTEFACTS}/")
