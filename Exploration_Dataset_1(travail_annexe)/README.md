# Annexe : exploration du jeu de factures fourni et début d'une chaîne OCR

Travail annexe du projet, présenté en annexe du rapport. On a exploré le premier jeu de données fourni avec
le sujet : 520 factures scannées, avec pour chaque image les zones annotées (GT) et les mots lus par un OCR,
avec leur indice de confiance. Ce n'est pas le dataset 1 du rapport (Kaggle) : comme il ne contient que
des factures, il ne sert pas à la classification.

On a ensuite commencé une chaîne de traitement OCR sur CPU : prétraitement léger, mesures de qualité de
l'image, détection des zones de texte (EAST) et choix du moteur OCR selon ces mesures.

## Contenu

    workspace/
    ├── notebooks/
    │   ├── 1-exploration/            chargement du jeu, visualisation GT / OCR, indices de confiance
    │   └── 2-construction_modele/    la chaîne OCR, une étape par dossier
    │       ├── step00_loading/              chargement, mesures et score de qualité des images
    │       ├── step01_preprocessing/        prétraitement léger, puis les mêmes mesures
    │       ├── step02_analyse_structurelle/ zones de texte détectées par EAST
    │       └── utils/                       fonctions des notebooks, et router_ocr_model.py (choix du moteur OCR)
    └── streamlit/                    application d'exploration du jeu (projet.py, pages/, contents/)

`streamlit/contents/cadrage/` contient le cadrage métier (constat, objectif, gains).

## Lancer

Tout tourne dans un conteneur Docker (Jupyter Lab sur http://localhost:8888), avec un fichier `.env` à la
racine de ce dossier :

    PROJECT_PREFIX="ml-ocr"
    IMAGE_VERSION="1.0"

Puis, depuis ce dossier :

    docker compose up -d --build

Les données ne sont pas sur GitHub. Les notebooks lisent l'archive fournie dans
`workspace/data/jdd/ImageAndXML_Data.zip` et le modèle EAST d'OpenCV dans
`workspace/models/east/frozen_east_text_detection.pb`. Le notebook `02_dataviz_GT_OCR` enregistre les
fichiers lus par l'application Streamlit dans `workspace/streamlit/data/raw/`.
