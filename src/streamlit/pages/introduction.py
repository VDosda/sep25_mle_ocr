import streamlit as st

import outils

st.title("Classification et extraction d'information sur un document")
st.markdown("""
Projet OCR | MLE | CONTINU | SEPT25

### Objectif
Classer automatiquement des documents administratifs scannés à partir de leur image, sans métadonnée et sans nom de fichier informatif.

### Cadrage métier
""")
st.image("src/streamlit/images/cadrage_metier.png", width=outils.LARGEUR_FIGURE)
st.markdown("""
Chaque document devient un actif numérique capable d’alimenter les processus métier, améliorer la qualité de décision
et renforcer la performance opérationnelle.

### Problématique
Classification multi-classe. Un document porte deux types d’information :
- son texte : le vocabulaire (« invoice », « total », « subject »…) ;
- sa mise en page : tableaux, colonnes, en-têtes, zones vides.

Laquelle de ces deux sources permet le mieux de reconnaître un document et gagne-t-on à les combiner ?

Toutes nos lignes de code sont disponibles sur :
github.com/DataScientest-Studio/sep25_mle_ocr

""")
