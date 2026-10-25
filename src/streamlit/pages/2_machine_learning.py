import pandas as pd
import streamlit as st

import outils

st.header("Machine Learning : l'image contre le texte")
st.markdown("Dataset 2 : de vrais scans, 6 classes. 1er extrait : 30 000 images, dont 6 000 pour le test.")


st.sidebar.divider()
documents = outils.load_ocr_sample()
classe = st.sidebar.selectbox("Classe du document montré", outils.CLASSES, index=outils.CLASSES.index("invoice"))
documents = documents[documents["label"] == classe]
numero = st.sidebar.slider("Numéro du document", 1, len(documents), 1)
document = documents.iloc[numero - 1]

pixels, mesures, texte, cosinus_ou_svm, langue, extrait_final = st.tabs(
    ["Pixels", "85 mesures", "Texte (OCR)", "Cosinus ou SVM", "Langue de l'OCR", "60 000 images"])

with pixels:
    st.markdown("""
- HOG : 50,6 %, à peine mieux que les 6 mesures globales (48,2 %). Le hasard : 16,7 %
- Pixels bruts : toutes les pages se ressemblent à plus de 99 % au sens du cosinus, le fond blanc domine
- Vérifié sur l'extrait de 60 000 images : 44,7 %
""")
    st.pyplot(outils.plot_steps(pd.DataFrame([
        ["Forme (format + encre)", 27.5, "centroïdes + cosinus"],
        ["Pixels centrés", 36.3, "centroïdes + cosinus"],
        ["Profils de projection", 43.0, "centroïdes + cosinus"],
        ["Pixels bruts (64 × 64)", 44.4, "centroïdes + cosinus"],
        ["6 mesures globales", 48.2, "XGBoost"],
        ["HOG", 50.6, "centroïdes + cosinus"],
    ], columns=["représentation", "accuracy (%)", "décision"])), width=outils.LARGEUR_GRAPHIQUE)

with mesures:
    st.markdown("""
- 70 mesures sur les pixels : luminosité, contraste, format, part d'encre, grille 8 × 8, traits de tableau
- 15 mesures de mise en page tirées de l'OCR, sans les mots : mots, lignes, blocs, taille des caractères, marges, confiance
- XGBoost : 77,7 % sur les 2 394 documents de test du modèle texte, 80,5 % des factures reconnues
""")
    st.image("src/streamlit/images/importance_mesures_xgboost.png", width=outils.LARGEUR_GRAPHIQUE)

with texte:
    st.markdown("""
- L'OCR prend environ une seconde par image : 1 800 documents (300 par classe), dont 360 pour le test
- Même méthode des centroïdes + cosinus : le texte fait bien mieux que l'image

| Source | Représentation | Décision | Accuracy |
|---|---|---|---|
| Image | HOG | Cosinus | 50,6 % |
| Texte (OCR) | TF-IDF mots, sans mots vides | Cosinus | 77,5 % |
| Texte (OCR) | TF-IDF caractères | Cosinus | 80,6 % |
""")
    colonne_image, colonne_texte = st.columns(2)
    colonne_image.image(document["filepath"], width=300)
    colonne_texte.markdown("##### Texte lu par l'OCR")
    colonne_texte.code(document["text"], language=None, wrap_lines=True)

with cosinus_ou_svm:
    st.markdown("""
- Retirer les mots vides ne change presque rien
- Mélanger mots et caractères : 2 documents de plus sur 360
- K-means, de 1 à 5 centroïdes par classe : de 78,9 % à 81,9 %
- Sur les mêmes vecteurs TF-IDF caractères, un SVM entraîné : 86,9 %, notre meilleur résultat
""")
    st.pyplot(outils.plot_steps(pd.DataFrame([
        ["TF-IDF mots", 76.9, "centroïdes + cosinus"],
        ["TF-IDF mots, sans mots vides", 77.5, "centroïdes + cosinus"],
        ["TF-IDF caractères, sans mots vides", 79.7, "centroïdes + cosinus"],
        ["TF-IDF caractères", 80.6, "centroïdes + cosinus"],
        ["Mélange mots + caractères", 81.1, "centroïdes + cosinus"],
        ["Caractères, 3 centroïdes par classe", 81.9, "centroïdes + cosinus"],
        ["TF-IDF caractères + SVM", 86.9, "SVM entraîné"],
    ], columns=["représentation", "accuracy (%)", "décision"])), width=outils.LARGEUR_GRAPHIQUE)
    st.markdown("##### Le SVM donne un poids à chaque morceau de mot : les plus importants pour chaque classe")
    st.pyplot(outils.plot_svm_ngrams(), width=outils.LARGEUR_FIGURE)

with langue:
    st.markdown("""
- L'OCR était réglé en français, alors que tous les documents sont en anglais
- Corrigé (lang="eng") : 135 mots lus par document au lieu de 123
- SVM sur TF-IDF caractères : 83,9 % → 86,9 %, soit 11 documents de plus sur 360
""")
    colonne_image, colonne_fra, colonne_eng = st.columns(3)
    colonne_image.image(document["filepath"])
    colonne_fra.markdown("##### Lu en français")
    colonne_fra.markdown(outils.highlight_words(document["text_fra"], document["text"], "red"))
    colonne_eng.markdown("##### Lu en anglais")
    colonne_eng.markdown(outils.highlight_words(document["text"], document["text_fra"], "green"))
    st.caption("Surlignés : les mots lus d'une seule des deux façons.")

with extrait_final:
    st.markdown("""
- Ré-entraîné sur l'extrait de 60 000 images : OCR de 12 000 documents (2 000 par classe)
- Sans les documents où l'OCR ne lit rien : 9 572 pour l'entraînement, 2 394 pour le test
- SVM sur TF-IDF caractères : 86,7 %, notre modèle texte de référence
""")
    st.pyplot(outils.plot_steps(pd.DataFrame([
        ["Dataset 1 (33 documents de test)", 93.9, "3 classes"],
        ["Dataset 2, 1er extrait (360)", 84.4, "6 classes"],
        ["Dataset 2, extrait final (2 394)", 86.7, "6 classes"],
    ], columns=["corpus", "accuracy (%)", "classes"])), width=outils.LARGEUR_GRAPHIQUE)
    st.markdown("""
    ### Deux modèles, mesurés sur les mêmes 2 394 documents
    - Texte : TF-IDF caractères + SVM, 86,7 %
    - Image : 85 mesures + XGBoost, 77,7 %
    """)
