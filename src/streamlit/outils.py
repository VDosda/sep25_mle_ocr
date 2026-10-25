"""Calculs de l'application Streamlit : données, modèles, prédictions, Grad-CAM, figures.

Le code reprend celui des notebooks 2.2, 4.3, 4.5.2, 4.5.3 et 5.
"""
import os
import sys

# sinon torch et XGBoost plantent ensemble sur macOS
os.environ["OMP_NUM_THREADS"] = "1"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers.utils import logging

logging.disable_progress_bar()
plt.rcParams["font.size"] = 13  # taille du texte des graphiques (10 par défaut)

# on se place à la racine du dépôt pour que les chemins "data/..." et "models/..." marchent
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

# pour la Démo 3 : l'OCR et les 85 mesures
sys.path.append(os.path.abspath("src/features"))
from ocr import read_words, words_to_text
from rvl_cdip_n import layout_features, pixel_features

DOSSIERS = {"Dataset 1 (165 images, 3 classes)": "data/document-classification-dataset_1",
            "Dataset 2 (60 000 images, 6 classes)": "data/rvl_cdip_filtered_dataset_3"}
ARTEFACTS = "data/artifacts_rvl_cdip_filtered_dataset_3"
# les 1 800 documents lus par l'OCR en 4.2, en anglais et en français
ECHANTILLON_OCR = "data/artifacts_rvl_cdip_filtered_dataset_2"
# Les documents des démos : nos modèles n'ont été entraînés sur aucun d'eux
JEUX_DE_TEST = {"Dataset 1 (Kaggle)": "data/artifacts_document-classification-dataset_1",
                "RVL-CDIP-N (documents récents du web)": "data/artifacts_rvl_cdip_n"}
DOSSIER_MODELES = "models"  # rempli par entrainer_modeles.py
DOSSIER_TELEVERSEMENTS = "src/streamlit/televersements"  # les images déposées sur la page Démo 3
CLASSES = ["email", "form", "invoice", "presentation", "resume", "scientific_report"]
# Notre modèle n'a pas scientific_publication : on la compte comme sa classe la plus proche, pour les deux modèles
CORRESPONDANCES = {"scientific_publication": "scientific_report"}

# Les noms affichés dans l'application
MODELES = {"image": "Image : 85 mesures + XGBoost",
           "cnn": "CNN : EfficientNet-B0",
           "texte": "Texte : TF-IDF caractères + SVM"}
STACKINGS = {"Stacking texte + image": ["texte", "image"],
             "Stacking texte + image + CNN": ["texte", "image", "cnn"]}
TOUS_LES_MODELES = list(MODELES.values()) + list(STACKINGS)
NOTRE_MODELE = "Notre stacking (texte + image + CNN)"
MODELE_HF = "DiT de Hugging Face (entraîné sur RVL-CDIP)"

# largeur des images et des graphiques, en pixels
LARGEUR_GRAPHIQUE = 600  # un graphique seul
LARGEUR_FIGURE = 850  # plusieurs graphiques ou documents côte à côte


# Modèles

if not os.path.exists(f"{DOSSIER_MODELES}/modele_texte.joblib"):
    raise FileNotFoundError("Modèles absents. Lancer une fois : .env/bin/python src/models/entrainer_modeles.py")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # la carte graphique du Mac

# CNN (partie 4.3) : on recrée l'architecture, puis on charge les poids enregistrés par le notebook 4.3_cnn
cnn = timm.create_model("efficientnet_b0", pretrained=False, num_classes=6)
cnn.load_state_dict(torch.load(f"{ARTEFACTS}/etape5_cnn_model.pt", map_location="cpu"))
cnn = cnn.to(device).eval()

# Modèle texte, modèle image et méta-modèle du stacking (partie 4.5) : enregistrés par entrainer_modeles.py
modele_texte = joblib.load(f"{DOSSIER_MODELES}/modele_texte.joblib")
modele_image = joblib.load(f"{DOSSIER_MODELES}/modele_image.joblib")
meta_modele = joblib.load(f"{DOSSIER_MODELES}/meta_modele.joblib")

# DiT de Hugging Face, déjà entraîné sur les 16 classes de RVL-CDIP (copie locale, sans internet)
processeur_dit = AutoImageProcessor.from_pretrained(f"{DOSSIER_MODELES}/dit-base-finetuned-rvlcdip")
dit = AutoModelForImageClassification.from_pretrained(f"{DOSSIER_MODELES}/dit-base-finetuned-rvlcdip").to(device).eval()
CLASSES_DIT = [dit.config.id2label[i].lower().replace(" ", "_") for i in range(len(dit.config.id2label))]

# Le même pré-traitement des images qu'à l'entraînement du CNN
normalisation = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # comme ImageNet
transform_eval = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(), normalisation])


# Données

@st.cache_data
def list_images(dossier):
    """Une ligne par image : son chemin et sa classe (= nom du sous-dossier)."""
    lignes = []
    for racine, _, fichiers in os.walk(dossier):
        for fichier in fichiers:
            if fichier.endswith(".png"):
                lignes.append({"filepath": os.path.join(racine, fichier),
                               "label": os.path.basename(racine)})
    return pd.DataFrame(lignes)


@st.cache_data
def load_probas(ensemble):
    """Probabilités des 3 modèles : un tableau par modèle, une ligne par document, une colonne par classe."""
    probas = {m: pd.read_parquet(f"{ARTEFACTS}/probas_{m}_{ensemble}.parquet").set_index("filepath")
              for m in MODELES}
    return {m: probas[m].loc[probas["texte"].index] for m in MODELES}  # mêmes documents, même ordre


@st.cache_data
def load_predictions():
    """Une ligne par document du test : classe réelle, classe prédite par chaque modèle et chaque stacking."""
    oof, test = load_probas("oof"), load_probas("test")
    df = pd.DataFrame({"filepath": test["texte"].index})
    df["label"] = df["filepath"].str.split("/").str[-2]  # la classe est le nom du dossier de l'image
    for m, nom in MODELES.items():
        df[nom] = np.array(CLASSES)[test[m].to_numpy().argmax(axis=1)]

    # Stacking (parties 4.5.2 et 4.5.3) : une régression logistique entraînée sur les probabilités des modèles
    y_train = oof["texte"].index.str.split("/").str[-2]
    for nom, modeles in STACKINGS.items():
        meta = LogisticRegression(max_iter=2000, random_state=42)
        meta.fit(np.hstack([oof[m] for m in modeles]), y_train)
        probas = meta.predict_proba(np.hstack([test[m] for m in modeles]))
        df[nom] = meta.classes_[probas.argmax(axis=1)]
    return df


def results_table(predictions):
    accuracies = [(predictions[nom] == predictions["label"]).mean() * 100 for nom in TOUS_LES_MODELES]
    return pd.DataFrame({"accuracy (%)": accuracies}, index=TOUS_LES_MODELES).round(1)


def accuracy_by_class(predictions, modeles):
    """Part (en %) des documents de chaque classe bien classés par chaque modèle, sur le test."""
    juste = {nom: predictions[nom] == predictions["label"] for nom in modeles}
    return (pd.DataFrame(juste).groupby(predictions["label"]).mean() * 100).round(1)


@st.cache_data
def load_test_documents(artefacts):
    """Les documents d'un jeu de test (un dossier de JEUX_DE_TEST) : chemin, classe réelle, texte lu par l'OCR et 85 mesures."""
    documents = pd.read_csv(f"{artefacts}/ocr_corpus.csv", keep_default_na=False)
    pixels = pd.read_parquet(f"{artefacts}/image_features.parquet")
    mise_en_page = pd.read_parquet(f"{artefacts}/layout_features.parquet").drop(columns="label")
    mesures = pixels.merge(mise_en_page, on="doc_id").rename(columns={"doc_id": "filepath"})
    documents["label"] = documents["label"].replace(CORRESPONDANCES)
    return documents.merge(mesures, on="filepath")


@st.cache_data
def load_ocr_sample():
    """Les 1 800 documents de la partie 4.2 : chemin, classe, texte lu par l'OCR en anglais (text) et en français (text_fra)."""
    documents = pd.read_csv(f"{ECHANTILLON_OCR}/ocr_echantillon_1800.csv", keep_default_na=False)
    en_francais = pd.read_csv(f"{ECHANTILLON_OCR}/ocr_echantillon_1800_fra.csv", keep_default_na=False)
    documents["text_fra"] = en_francais["text"]  # mêmes documents, même ordre
    return documents


@st.cache_data
def svm_ngrams(n=8):
    """Pour chaque classe, les n morceaux de mots qui pèsent le plus dans le modèle texte, avec leur poids."""
    morceaux = modele_texte[0].named_transformers_["tfidf"].get_feature_names_out()
    poids = np.mean([c.estimator.coef_ for c in modele_texte[-1].calibrated_classifiers_], axis=0)
    resultat = {}
    for classe, poids_classe in zip(modele_texte.classes_, poids):
        gardes = {}
        for i in np.argsort(poids_classe)[::-1]:  # du plus grand poids au plus petit
            morceau = morceaux[i].strip()
            if sum(c.isalpha() for c in morceau) >= 3 and not any(morceau in m or m in morceau for m in gardes):
                gardes[morceau] = poids_classe[i]
            if len(gardes) == n:
                break
        resultat[classe] = gardes
    return resultat


def highlight_words(texte, autre_texte, couleur):
    """Le texte en Markdown, avec surlignés les mots absents de l'autre texte (deux lectures du même document)."""
    autres_mots = set(autre_texte.split())
    mots = []
    for mot in texte.split():
        mot_affiche = "".join("\\" + c if c in "\\`*_{}[]()#+-.!|$<>~:" else c for c in mot)  # caractères spéciaux du Markdown
        mots.append(mot_affiche if mot in autres_mots else f":{couleur}-background[{mot_affiche}]")
    return " ".join(mots)


def load_image(filepath):
    return Image.open(filepath).convert("RGB")


# Images déposées sur la page Démo 3

def save_upload(fichier):
    """Enregistre une image déposée sur la page comme les documents de RVL-CDIP : noir et blanc, 1 000 pixels de haut."""
    image = Image.open(fichier).convert("L")
    image = image.resize((round(image.width * 1000 / image.height), 1000))
    os.makedirs(DOSSIER_TELEVERSEMENTS, exist_ok=True)
    chemin = f"{DOSSIER_TELEVERSEMENTS}/{fichier.file_id}.png"
    image.save(chemin)
    return chemin


@st.cache_data
def prepare_document(filepath):
    """Ce que nos modèles lisent d'une image : le texte lu par l'OCR et les 85 mesures (une ligne, comme load_test_documents)."""
    boites = read_words((filepath, "", "eng"))  # une ligne par mot lu : sa boîte, sa confiance, son texte
    document = pd.DataFrame({"filepath": [filepath], "text": [words_to_text(boites).get(filepath, "")]})
    mesures = pd.DataFrame([pixel_features(filepath)]).join(layout_features(boites, [filepath]).reset_index(drop=True))
    return pd.concat([document, mesures], axis=1)


# Prédictions

def predict_cnn(filepaths):
    """Les images passent dans le CNN, par lots de 16 : probabilité de chacune de nos 6 classes."""
    probas = []
    with torch.no_grad():
        for debut in range(0, len(filepaths), 16):
            x = torch.stack([transform_eval(load_image(f)) for f in filepaths[debut:debut + 16]]).to(device)
            probas.append(torch.softmax(cnn(x), dim=1).cpu().numpy())
    return np.concatenate(probas)


@st.cache_data
def predict_dit(filepaths):
    """Les images passent dans le DiT, par lots de 16 : la classe prédite (parmi ses 16) pour chacune."""
    classes = []
    with torch.no_grad():
        for debut in range(0, len(filepaths), 16):
            images = [load_image(f) for f in filepaths[debut:debut + 16]]
            logits = dit(**processeur_dit(images=images, return_tensors="pt").to(device)).logits
            classes += [CLASSES_DIT[i] for i in logits.argmax(dim=1).tolist()]
    return [CORRESPONDANCES.get(c, c) for c in classes]



@st.cache_data
def predict_pipeline(documents):
    """Notre meilleur modèle, étape par étape : les 3 modèles, puis le méta-modèle qui combine leurs probabilités."""
    probas = {"texte": modele_texte.predict_proba(documents),  # lit la colonne "text"
              "image": modele_image.predict_proba(documents),  # lit les 85 colonnes de mesures
              "cnn": predict_cnn(list(documents["filepath"]))}
    # les 18 probabilités (6 par modèle), dans le même ordre qu'à l'entraînement du méta-modèle
    probas["stacking"] = meta_modele.predict_proba(np.hstack([probas["texte"], probas["image"], probas["cnn"]]))
    return {etape: pd.DataFrame(p, index=documents["filepath"], columns=CLASSES) for etape, p in probas.items()}


@st.cache_data
def predict_stacking(documents):
    """Notre meilleur modèle : la classe prédite par le stacking pour chaque document."""
    return predict_pipeline(documents)["stacking"].idxmax(axis=1).to_numpy()


@st.cache_data
def grad_cam(filepath, classe, layer_name="conv_head"):
    """Carte Grad-CAM d'un document pour une classe : 12 × 12 valeurs de 0 à 1 (1 = zone qui pèse le plus)."""
    x = transform_eval(load_image(filepath)).unsqueeze(0).to(device)

    # passe avant : le hook garde la sortie de la dernière couche de convolution
    saved = {}
    def forward_hook(module, input, output):
        saved["activations"] = output
    hook = cnn.get_submodule(layer_name).register_forward_hook(forward_hook)
    output = cnn(x)
    hook.remove()

    # passe arrière : gradients du score de la classe
    cnn.zero_grad()
    saved["activations"].retain_grad()
    output[0, CLASSES.index(classe)].backward()
    gradients, activations = saved["activations"].grad[0], saved["activations"][0]

    # un poids par filtre, somme pondérée, on garde le positif et on ramène entre 0 et 1
    pooled_grads = gradients.mean(dim=(1, 2))
    heatmap = torch.relu((pooled_grads[:, None, None] * activations).sum(dim=0))
    return (heatmap / (heatmap.max() + 1e-8)).detach().cpu().numpy()


# Figures

def plot_classes(df):
    fig = plt.figure(figsize=(7, 3))
    sns.countplot(x="label", data=df, order=sorted(df["label"].unique()))
    plt.xticks(rotation=20)
    plt.xlabel("")
    plt.ylabel("Nombre d'images")
    return fig


def plot_examples(df, classe, n):
    """n images de la classe, tirées au hasard."""
    exemples = df[df["label"] == classe].sample(n, random_state=42)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
    for ax, chemin in zip(axes, exemples["filepath"]):
        ax.imshow(Image.open(chemin).convert("L"), cmap="gray")
        ax.axis("off")
    return fig


def plot_steps(etapes):
    """Barres horizontales : l'accuracy de chaque modèle."""
    nom, accuracy, couleur = etapes.columns
    fig = plt.figure(figsize=(8, 4))
    sns.barplot(data=etapes, x=accuracy, y=nom, hue=couleur, dodge=False)
    for i, valeur in enumerate(etapes[accuracy]):
        plt.text(valeur + 1, i, f"{valeur} %", va="center")
    plt.xlim(0, 110)
    plt.ylabel("")
    plt.legend(title="", loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)  # légende au-dessus des barres
    return fig


def plot_svm_ngrams():
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    for ax, (classe, gardes) in zip(axes.ravel(), svm_ngrams().items()):
        ax.barh(list(gardes)[::-1], list(gardes.values())[::-1])  # le plus grand poids en haut
        ax.set_title(classe)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(predictions, modele):
    matrice = pd.crosstab(predictions["label"], predictions[modele], rownames=["Réel"], colnames=["Prédit"])
    fig = plt.figure(figsize=(6, 4))
    sns.heatmap(matrice, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{modele} : {(predictions[modele] == predictions['label']).mean():.1%}")
    return fig


def plot_confusion_matrices(matrices, classes):
    """Des matrices de confusion recopiées d'un notebook, côte à côte. matrices : {titre : lignes de la matrice}."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(3.5 * len(matrices) + 1.5, 3.5))
    for i, (ax, (titre, lignes)) in enumerate(zip(axes, matrices.items())):
        matrice = pd.DataFrame(lignes, index=classes, columns=classes)
        # les noms des classes réelles seulement à gauche de la première matrice
        sns.heatmap(matrice, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, yticklabels=(i == 0))
        ax.set_title(titre)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel" if i == 0 else "")
        ax.tick_params(axis="x", rotation=20)
        ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    return fig


def plot_cnn_history():
    """Courbes d'entraînement du CNN (enregistrées par le notebook 4.3_cnn)."""
    historique = pd.read_csv(f"{ARTEFACTS}/cnn_historique.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, mesure, titre in [(axes[0], "loss", "Perte (loss)"), (axes[1], "acc", "Accuracy")]:
        ax.plot(historique["epoch"], historique[f"train_{mesure}"], marker="o", label="train")
        ax.plot(historique["epoch"], historique[f"val_{mesure}"], marker="o", label="validation")
        ax.set_xlabel("epoch")
        ax.set_title(titre)
        ax.legend()
    return fig


def plot_grad_cam(filepath, classes):
    """Le document, puis une carte Grad-CAM du CNN par classe demandée."""
    image = load_image(filepath)
    probas = pd.Series(predict_cnn([filepath])[0], index=CLASSES)
    fig, axes = plt.subplots(1, 1 + len(classes), figsize=(4.5 * (1 + len(classes)), 6))
    axes[0].imshow(image)
    axes[0].set_title("Document")
    for ax, classe in zip(axes[1:], classes):
        ax.imshow(image)
        # la carte 12 × 12 est agrandie à la taille du document
        ax.imshow(grad_cam(filepath, classe), cmap="jet", alpha=0.45, interpolation="bilinear",
                  extent=(*ax.get_xlim(), *ax.get_ylim()))
        ax.set_title(f"Grad-CAM pour {classe} ({probas[classe]:.0%})")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    return fig
