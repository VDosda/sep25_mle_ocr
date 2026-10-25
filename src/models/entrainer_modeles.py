"""Entraîne et enregistre les modèles de la démo. À lancer une fois, depuis la racine du projet :

    .env/bin/python src/models/entrainer_modeles.py

Ce sont les modèles de la partie 4.5 (notebooks 4.5.2 et 4.5.3), entraînés sur le train de l'extrait final :
- modele_texte.joblib : TF-IDF caractères + SVM linéaire ;
- modele_image.joblib : 85 mesures + XGBoost ;
- meta_modele.joblib  : la régression logistique du stacking à 3 modèles (texte + image + CNN).
Le CNN est déjà enregistré par le notebook 4.3_cnn (etape5_cnn_model.pt).

SUR MAC : torch n'est pas importé ici, il plante avec l'entraînement de XGBoost.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

ARTEFACTS = "data/artifacts_rvl_cdip_filtered_dataset_3"
DOSSIER_MODELES = "models"
CLASSES = ["email", "form", "invoice", "presentation", "resume", "scientific_report"]

corpus = pd.read_csv(f"{ARTEFACTS}/ocr_corpus.csv", keep_default_na=False)
corpus = corpus[corpus["text"] != ""].reset_index(drop=True)
train, test = train_test_split(corpus, test_size=0.2, random_state=42, stratify=corpus["label"])

pixels = pd.read_parquet(f"{ARTEFACTS}/image_features.parquet")
mise_en_page = pd.read_parquet(f"{ARTEFACTS}/layout_features.parquet").drop(columns="label")
mesures = pixels.merge(mise_en_page, on="doc_id").set_index("doc_id")
MESURES = list(mesures.columns)
X_train = mesures.loc[train["filepath"]].assign(text=train["text"].to_numpy())
X_test = mesures.loc[test["filepath"]].assign(text=test["text"].to_numpy())

modele_texte = make_pipeline(
    ColumnTransformer([("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True), "text")]),
    CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=5000), cv=5))
modele_image = make_pipeline(
    ColumnTransformer([("mesures", "passthrough", MESURES)]),
    XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric="mlogloss"))

print("Entraînement du modèle texte...")
modele_texte.fit(X_train, train["label"])
print("Entraînement du modèle image...")
modele_image.fit(X_train, LabelEncoder().fit(CLASSES).transform(train["label"]))

oof = np.hstack([pd.read_parquet(f"{ARTEFACTS}/probas_{m}_oof.parquet").set_index("filepath").loc[train["filepath"], CLASSES]
                 for m in ["texte", "image", "cnn"]])
meta_modele = LogisticRegression(max_iter=2000, random_state=42).fit(oof, train["label"])

probas_texte = modele_texte.predict_proba(X_test)
probas_image = modele_image.predict_proba(X_test)
probas_cnn = pd.read_parquet(f"{ARTEFACTS}/probas_cnn_test.parquet").set_index("filepath").loc[test["filepath"], CLASSES]
print(f"Accuracy test — texte : {accuracy_score(test['label'], np.array(CLASSES)[probas_texte.argmax(axis=1)]):.1%}")
print(f"Accuracy test — image : {accuracy_score(test['label'], np.array(CLASSES)[probas_image.argmax(axis=1)]):.1%}")
print(f"Accuracy test — stacking : {meta_modele.score(np.hstack([probas_texte, probas_image, probas_cnn]), test['label']):.1%}")

os.makedirs(DOSSIER_MODELES, exist_ok=True)
joblib.dump(modele_texte, f"{DOSSIER_MODELES}/modele_texte.joblib")
joblib.dump(modele_image, f"{DOSSIER_MODELES}/modele_image.joblib")
joblib.dump(meta_modele, f"{DOSSIER_MODELES}/meta_modele.joblib")
print(f"Modèles enregistrés dans {DOSSIER_MODELES}/")
