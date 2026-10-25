import pandas as pd
import streamlit as st

import outils

st.header("Deep Learning et LLM : partir de l'image brute")
st.markdown("Mêmes documents de test que le modèle texte de référence (SVM, 86,7 %).")

extraction, fine_tuning, cnn_et_texte, llm = st.tabs(["Feature Extraction", "Fine-Tuning", "CNN et texte", "LLM"])

with extraction:
    st.markdown("""
- 3 réseaux pré-entraînés et gelés, une régression logistique sur le vecteur qu'ils sortent
- EfficientNet-B0, DiT-base et BEiT-base : presque égaux, autour de 76 %
- DiT déjà entraîné sur RVL-CDIP : 98,1 %, écarté (il a appris sur la base d'où viennent nos images)
""")
    st.pyplot(outils.plot_steps(pd.DataFrame([
        ["EfficientNet-B0 (CNN, ImageNet)", 75.8, "réseau gelé"],
        ["BEiT-base (ImageNet-22k)", 76.0, "réseau gelé"],
        ["DiT-base (documents)", 76.1, "réseau gelé"],
        ["Texte : TF-IDF caractères + SVM", 86.7, "référence"],
    ], columns=["modèle", "accuracy (%)", "type"])), width=outils.LARGEUR_GRAPHIQUE)

with fine_tuning:
    st.markdown("""
- EfficientNet-B0 : dernière couche remplacée par 6 classes, tout le réseau ré-entraîné
- Images en 384 × 384 pixels, pour que le texte reste lisible
- Augmentation légère (rotation, luminosité, contraste), pas de retournement
- 10 epochs au plus, arrêt anticipé sur une validation de 10 % du train
- Le train monte vers 98 %, la validation plafonne vers 86 % dès l'epoch 4 : début de surapprentissage
""")
    st.pyplot(outils.plot_cnn_history(), width=outils.LARGEUR_FIGURE)

with cnn_et_texte:
    st.markdown("""
- CNN : 85,1 %, à peine moins que le texte (86,7 %), mieux que nos 85 mesures (77,7 %)
- Il reconnaît mieux les e-mails, les formulaires et les présentations ; moins bien les factures, les CV et les rapports
- Il classe bien 194 des 319 documents où le texte se trompe
""")
    st.write("Part des documents bien classés, par classe (en %) :")
    st.dataframe(outils.accuracy_by_class(outils.load_predictions(), [outils.MODELES["texte"], outils.MODELES["cnn"]]))

with llm:
    st.markdown("""
- LLM multimodal en local (Gemma4), sans entraînement : on lui donne l'image et la liste des classes
- Plusieurs secondes par document : 10 images par classe
- Dataset 1 : 93,3 % (30 images)
- Dataset 2 : 73,3 % (60 images), loin du texte
- Mêmes erreurs que nos autres modèles : factures prises pour des formulaires, présentations pour des rapports
""")
