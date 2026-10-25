import streamlit as st

import outils

st.header("Données et démarche")
st.markdown("""
### Deux jeux de données

| | Dataset 1 | Dataset 2 |
|---|---|---|
| Source | Kaggle | Hugging Face (RVL-CDIP) |
| Classes | 3 : email, resume, scientific_publication | 6 : email, form, invoice, presentation, resume, scientific_report |
| Taille | 165 images (55 par classe) | 30 000 puis 60 000 images (10 000 par classe) |

- Dataset 1 : le jeu proposé, petit et propre
- Dataset 2 : extrait de RVL-CDIP (400 000 scans d'archives, 16 classes), souvent abîmés
- Classes équilibrées : accuracy et matrice de confusion
""")

st.sidebar.divider()
dataset = st.sidebar.selectbox("Jeu de données", list(outils.DOSSIERS))
df = outils.list_images(outils.DOSSIERS[dataset])
st.write(len(df), "images")
st.pyplot(outils.plot_classes(df), width=outils.LARGEUR_GRAPHIQUE)

classe = st.sidebar.selectbox("Exemples de la classe", sorted(df["label"].unique()))
n = st.sidebar.slider("Nombre d'exemples", 2, 6, 3)
st.pyplot(outils.plot_examples(df, classe, n), width=outils.LARGEUR_FIGURE)

st.markdown("""
### Démarche
1. Preuve de concept sur le dataset 1
2. Machine Learning : mesures de l'image, pixels, texte lu par OCR
3. Deep Learning (CNN) et LLM
4. Explicabilité du CNN (Grad-CAM, SHAP)
5. Stacking : combiner les modèles
""")
