import streamlit as st

from utils import (
    load_md, 
)
import pandas as pd

from customclass import (
    PolygonStatus,
    classify_confidence
)

from dashboard import (
    show_img_gt_ocr,
    show_synthese_ocr,
    show_stats_globales,
    show_export_outliers,
    show_heatmap_confiance,
    show_confiance_ocr,
)

st.title("Exploitation du dataset fourni")
st.subheader("Analyse détaillée ")

# Chargement des données
@st.cache_data
def load_datasets():

    df_1 = pd.read_pickle("data/raw/ImageAndXML_Data_Images.pkl")
    df_2 = pd.read_pickle("data/raw/ImageAndXML_Data_GT.pkl")
    df_3 = pd.read_pickle("data/raw/ImageAndXML_Data_OCR.pkl")
    df_3["confidence_level"] = df_3["confidence"].apply(classify_confidence)
    return df_1, df_2, df_3

df_images, df_gt, df_ocr = load_datasets()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Dataset",
    "2. Layout (GT)",
    "3. OCR",
    "4. Qualité des images",
    "5. Synthèse",
    "6. Visualisation"
])

with tab1:
    st.markdown(load_md("exploration/dataset.md"), unsafe_allow_html=True)

with tab2:
    st.markdown(load_md("exploration/layout.md"), unsafe_allow_html=True)

with tab3:
    st.markdown(load_md("exploration/ocr.md"), unsafe_allow_html=True)

with tab4:
    st.markdown(load_md("exploration/quality.md"), unsafe_allow_html=True)

with tab5:
    st.markdown(load_md("exploration/synthese.md"), unsafe_allow_html=True)

with tab6:
    tab_visu, tab_confiance, tab_synthese, tab_stats, tab_export, tab_heatmap = st.tabs([
        "Image GT & OCR",
        "Confiance OCR",
        "Synthèse OCR",
        "Statistiques globales",
        "Export des outliers",
        "Heatmap confiance"
    ])

    # Onglet 1 : Visualisation GT / OCR
    with tab_visu:
        show_img_gt_ocr(df_images, df_gt, df_ocr)

    # Onglet 2 : Confiance OCR
    with tab_confiance:
        show_confiance_ocr(df_ocr)

    # Onglet 3 : Synthèse OCR
    with tab_synthese:
        show_synthese_ocr(df_ocr)

    # Onglet 4 : Statistiques globales
    with tab_stats:
        show_stats_globales(df_ocr)

    # Onglet 5 : Export des outliers
    with tab_export:
        show_export_outliers(df_ocr)

    # Onglet 6 : Heatmap confiance
    with tab_heatmap:
        show_heatmap_confiance(df_ocr)

