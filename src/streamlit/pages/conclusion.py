import pandas as pd
import streamlit as st

import outils

st.header("Conclusion")
st.markdown("### Texte ou mise en page ?")

etapes = pd.DataFrame([
    ["6 mesures globales + XGBoost", 48.2, "1er extrait (30 000 images)"],
    ["HOG + cosinus", 50.6, "1er extrait (30 000 images)"],
    ["LLM seul, sans entraînement", 73.3, "1er extrait (30 000 images)"],
    ["TF-IDF caractères + cosinus", 80.6, "1er extrait (30 000 images)"],
    ["Image : 85 mesures + XGBoost", 77.7, "Extrait final (60 000 images)"],
    ["CNN EfficientNet-B0", 85.1, "Extrait final (60 000 images)"],
    ["Texte : TF-IDF caractères + SVM", 86.7, "Extrait final (60 000 images)"],
    ["Stacking texte + image", 89.5, "Extrait final (60 000 images)"],
    ["Stacking texte + image + CNN", 91.9, "Extrait final (60 000 images)"],
], columns=["modèle", "accuracy (%)", "mesuré sur"])
st.pyplot(outils.plot_steps(etapes), width=outils.LARGEUR_GRAPHIQUE)

st.markdown("""

- La mise en page fonctionne : 85,1 % avec le CNN
- Le texte reste la source la plus forte : 86,9 % contre 50,6 % pour les pixels (1er extrait)
- Les combiner donne notre meilleur modèle : **91,9 %**, et 95 % sur RVL-CDIP-N
- Modèle retenu : le stacking texte + image + CNN, 91,9 %

### Perspectives
- Extraire les informations clés : fournisseur, date, montant total, pour des traitements automatiques et des statistiques
- Indexer les documents (Elasticsearch : BM25 et recherche kNN) : texte lu par l'OCR, mise en page, structure
- S'appuyer sur cet index pour consolider la classification, et confier les cas non tranchés à un LLM multimodal
- Alimenter un index de données propres : anonymisation, chiffrement, nettoyage, dédoublonnage
""")
