import streamlit as st
import plotly.graph_objects as go
import math

from customclass import (
    classify_confidence,
)

def show_confiance_ocr(df_ocr):

    st.markdown("### Analyse de la confiance OCR")

    # Pagination
    page_size = 20
    images = sorted(df_ocr["image_filename"].unique())
    num_pages = math.ceil(len(images) / page_size)

    # Sélecteur de page
    page = st.number_input(
        "Page",
        min_value=0,
        max_value=num_pages - 1,
        value=0,
        step=1
    )

    # Sélecteur du seuil
    seuil = st.slider(
        "Seuil de confiance",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )

    # Extraction des données de la page
    start = page * page_size
    end = start + page_size
    imgs = images[start:end]
    df_page = df_ocr[df_ocr["image_filename"].isin(imgs)]

    # Construction du graphique
    fig = go.Figure()

    # 1) Boxplot
    fig.add_trace(go.Box(
        x=df_page["image_filename"],
        y=df_page["confidence"],
        boxpoints=False,
        name="Confidence",
        marker_color="lightgray"
    ))

    # 2) Outliers
    outliers = df_page[df_page["confidence"] < seuil]
    non_outliers = df_page[df_page["confidence"] >= seuil]

    # Points non-outliers
    fig.add_trace(go.Scatter(
        x=non_outliers["image_filename"],
        y=non_outliers["confidence"],
        mode="markers",
        marker=dict(size=6, color="lightgray"),
        hoverinfo="skip",
        name="Non-outliers"
    ))

    # Points outliers colorés
    fig.add_trace(go.Scatter(
        x=outliers["image_filename"],
        y=outliers["confidence"],
        mode="markers",
        marker=dict(
            size=9,
            color=outliers["confidence"],
            colorscale="RdYlGn",
            showscale=True
        ),
        text=[
            f"Image: {row['image_filename']}<br>"
            f"Confidence: {row['confidence']:.3f}<br>"
            f"Texte: {row['text']}<br>"
            f"Word ID: {row['word_id']}"
            for _, row in outliers.iterrows()
        ],
        hovertemplate="%{text}<extra></extra>",
        name="Outliers"
    ))

    fig.update_layout(
        title=f"Confidence OCR — page {page+1}/{num_pages} — seuil={seuil}",
        xaxis_title="Image",
        yaxis_title="Confidence",
        xaxis_tickangle=45,
        height=700
    )

    # Affichage
    st.plotly_chart(fig, use_container_width=True)
