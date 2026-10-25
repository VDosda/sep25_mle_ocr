import streamlit as st

from utils import load_md

# Configuration générale de l'application
st.set_page_config(
    page_title="Module MLE - OCR & Classification",
    layout="wide"
)

# Page d'accueil
st.title("Module MLE - Projet OCR & Classification de documents")
st.subheader("Plateforme IA pour l’exploitation intelligente des documents scannés")

# Contenu Markdown de l'accueil
st.markdown(
    load_md("accueil.md"),
    unsafe_allow_html=True
)
