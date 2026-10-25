import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def list_images(dossier):
    lignes = []
    for racine, _, fichiers in os.walk(dossier):
        for fichier in fichiers:
            if fichier.endswith(".png"):
                lignes.append({"filepath": os.path.join(racine, fichier),
                               "label": os.path.basename(racine)})
    return pd.DataFrame(lignes)


def classify_by_cosine(X_train, y_train, X_test):
    if hasattr(X_train, "toarray"):  # le TF-IDF arrive en matrice creuse
        X_train = X_train.toarray()
    y_train = np.asarray(y_train)
    classes = sorted(set(y_train))
    centroides = normalize(np.vstack([X_train[y_train == classe].mean(axis=0) for classe in classes]))
    similarites = pd.DataFrame(cosine_similarity(X_test, centroides), columns=classes)
    return similarites.idxmax(axis=1).values, similarites
