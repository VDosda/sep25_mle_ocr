import pandas as pd
import streamlit as st

import outils

st.header("Démo : testez notre modèle avec vos documents")
st.markdown("""
Déposez une ou plusieurs images. Chacune passe dans toute la chaîne : OCR, 85 mesures, 3 modèles, stacking.

Notre modèle choisit parmi ses 6 classes : `email`, `form`, `invoice`, `presentation`, `resume`,
`scientific_report`.
""")

fichiers = st.file_uploader("Vos documents", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)

for fichier in fichiers:
    st.divider()
    filepath = outils.save_upload(fichier)
    document = outils.prepare_document(filepath)

    # Vraies prédictions
    probas = outils.predict_pipeline(document)
    tableau = pd.DataFrame({etape: probas[etape].iloc[0] for etape in probas})
    predictions = tableau.idxmax()  # pour chaque étape, la classe la plus probable

    gauche, droite = st.columns([1, 2])
    gauche.image(filepath, caption=fichier.name)
    with droite:
        st.markdown(f"### {outils.NOTRE_MODELE} : **{predictions['stacking']}** ({tableau['stacking'].max():.0%})")
        for m, nom in outils.MODELES.items():
            st.markdown(f"- {nom} : **{predictions[m]}** ({tableau[m].max():.0%})")
        st.dataframe(tableau.style.format("{:.0%}").background_gradient(cmap="Blues", vmin=0, vmax=1))
        st.markdown(f"Pour comparaison, le {outils.MODELE_HF} prédit : **{outils.predict_dit([filepath])[0]}**")
        texte = document["text"].iloc[0]
        st.caption(f"Texte lu par l'OCR : {texte[:300]}…" if texte else "L'OCR n'a lu aucun texte.")
