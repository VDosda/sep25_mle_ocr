import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def list_images(dossier):
    lignes = []
    for racine, _, fichiers in os.walk(dossier):
        for fichier in fichiers:
            if fichier.endswith(".png"):
                lignes.append({"filepath": os.path.join(racine, fichier),
                               "label": os.path.basename(racine)})
    return pd.DataFrame(lignes)


def add_visual_metrics(df):
    lignes = []
    for chemin in tqdm(df["filepath"]):
        image = Image.open(chemin).convert("L")
        pixels = np.array(image, dtype=np.float32)
        largeur, hauteur = image.size
        lignes.append({"width": largeur,
                       "height": hauteur,
                       "brightness": pixels.mean(),
                       "contrast": pixels.std(),
                       "dark_ratio": (pixels < 128).mean(),  # pixels sombres
                       "aspect_ratio": largeur / hauteur})
    return pd.concat([df, pd.DataFrame(lignes)], axis=1)


def values_by_class(df, colonne):
    """une série par classe, comme le veulent les tests de scipy"""
    return [df.loc[df["label"] == classe, colonne] for classe in sorted(df["label"].unique())]


def show_mosaic(df, n_par_classe=4):
    classes = sorted(df["label"].unique())
    fig, axes = plt.subplots(len(classes), n_par_classe,
                             figsize=(3 * n_par_classe, 3.5 * len(classes)))
    for i, classe in enumerate(classes):
        exemples = df[df["label"] == classe].sample(n_par_classe, random_state=42)
        for j, chemin in enumerate(exemples["filepath"]):
            axes[i, j].imshow(Image.open(chemin).convert("L"), cmap="gray")
            axes[i, j].set_title(classe)
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.show()
