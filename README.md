Classification et extraction d'information sur un document
==============================

Projet de la formation Machine Learning Engineer de DataScientest (promotion septembre 2025).
Isabelle WUILBEAUX, Sophie BONNEAU, Victor DOSDA.

On classe des documents scannés (email, facture, formulaire, CV...) à partir de leur image.
On compare le texte lu par l'OCR, la mise en page et un CNN, puis on combine les trois (stacking).
Notre meilleur modèle, le stacking texte + image + CNN, obtient 91,9 % d'accuracy sur le test du dataset 2.

Données
------------

Les données ne sont pas sur GitHub (plusieurs Go). Elles vont dans `data/` :

- dataset 1 : 165 images, 3 classes (Kaggle), dans `data/document-classification-dataset_1/`
- dataset 2 : deux extraits de RVL-CDIP (`aharley/rvl_cdip` sur Hugging Face), 6 classes,
  dans `data/rvl_cdip_filtered_dataset_2/` (30 000 images) et `data/rvl_cdip_filtered_dataset_3/` (60 000 images)
- RVL-CDIP-N : le jeu de test de l'application, préparé par `src/features/rvl_cdip_n.py`

Dans chaque dossier, un sous-dossier par classe : son nom sert d'étiquette.
Le texte lu par l'OCR est calculé une seule fois pour tous les documents par `src/features/ocr.py`
(environ 25 minutes).

Installation
------------

Python 3.12 et Tesseract. Sur Mac uniquement, Tesseract s'installe avec Homebrew :
`brew install tesseract tesseract-lang`. Sur un autre système, l'installer avec son gestionnaire de paquets.
Puis :

```bash
python3.12 -m venv .env
.env/bin/pip install -r requirements.txt
.env/bin/python -c "import nltk; nltk.download('stopwords')"
```

Organisation du projet
------------

    ├── Exploration_Dataset_1(travail_annexe)
    │                      <- annexe : exploration du jeu de factures fourni et début d'une chaîne OCR
    ├── LICENSE
    ├── README.md
    ├── requirements.txt   <- les bibliothèques Python, avec les versions utilisées
    ├── data               <- les données, sur l'ordinateur mais pas sur GitHub (dans le .gitignore)
    ├── models             <- les modèles enregistrés pour l'application (pas sur GitHub, trop lourds)
    ├── notebooks          <- un notebook par partie du rapport, numéroté comme lui (2.1, 3.2, 4.5.3...)
    │                         outils_2.py, outils_3.py, outils_4.py : les fonctions communes de chaque partie
    ├── reports            <- le rapport
    └── src
        ├── features       <- ocr.py : l'OCR de tous les documents, à lancer une fois
        │                     rvl_cdip_n.py : prépare le jeu de test RVL-CDIP-N
        ├── models         <- entrainer_modeles.py : entraîne et enregistre les modèles de l'application
        └── streamlit      <- l'application de la soutenance : streamlit_app.py, outils.py, pages/

Certains notebooks utilisent les résultats d'autres notebooks : lancer 4.2_texte_svm_60k avant
4.3_feature_extraction et 4.5.1_meta_modele, puis 4.3_cnn et 4.5.2 avant 4.5.3.

Lancer l'application
------------

Depuis la racine du dépôt, la première fois : entraîner les modèles de l'application, puis copier le DiT
de Hugging Face dans `models/`.

```bash
.env/bin/python src/models/entrainer_modeles.py
.env/bin/python -c "from transformers import AutoImageProcessor, AutoModelForImageClassification; n = 'microsoft/dit-base-finetuned-rvlcdip'; [m.from_pretrained(n).save_pretrained('models/dit-base-finetuned-rvlcdip') for m in (AutoImageProcessor, AutoModelForImageClassification)]"
```

Puis :

```bash
.env/bin/streamlit run src/streamlit/streamlit_app.py
```

L'application s'ouvre sur http://localhost:8501.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
