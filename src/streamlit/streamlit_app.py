import matplotlib.pyplot as plt
import streamlit as st

import outils

st.set_page_config(layout="wide")
plt.close("all")


pages = [
    st.Page("pages/introduction.py", title="Introduction"),
    st.Page("pages/donnees_et_demarche.py", title="Données et démarche"),
    st.Page("pages/1_preuve_de_concept.py", title="1. Preuve de concept"),
    st.Page("pages/2_machine_learning.py", title="2. Machine Learning"),
    st.Page("pages/3_deep_learning_et_llm.py", title="3. Deep Learning et LLM"),
    st.Page("pages/4_explicabilite.py", title="4. Explicabilité"),
    st.Page("pages/5_stacking.py", title="5. Stacking"),
    st.Page("pages/demo_stacking_pas_a_pas.py", title="Démo : le Stacking pas à pas"),
    st.Page("pages/demo_face_au_dit_de_hugging_face.py", title="Démo : face au DiT de Hugging Face"),
    st.Page("pages/conclusion.py", title="Conclusion"),
]

pages_cachees = [st.Page("pages/demo_3.py", title="Démo 3"),
                 st.Page("pages/cas_difficiles.py", title="Cas difficiles")]
navigation = st.navigation(pages + pages_cachees, position="hidden")

st.sidebar.title("Projet OCR MLE 2025")
st.sidebar.write("Isabelle WUILBEAUX, Sophie BONNEAU, Victor DOSDA")
for page in pages:
    st.sidebar.page_link(page)

navigation.run()