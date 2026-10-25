import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import base64
import numpy as np
from io import BytesIO
from PIL import Image

from customclass import (
    PolygonStatus,
)

def add_image_trace(img_b64, fullscreen=False):
    img_bytes = base64.b64decode(img_b64)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    img_np = np.array(img)

    fig = px.imshow(img_np)
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(visible=False, range=[0, w])
    fig.update_yaxes(visible=False, range=[h, 0], scaleanchor='x', scaleratio=1)

    # plein écran automatique
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=1100 if fullscreen else 700
    )

    return fig


def add_gt_trace(fig, df_gt_img):
    for _, row in df_gt_img.iterrows():
        if row["polygon_status"] == PolygonStatus.VALIDE:
            x, y = row["polygon"].exterior.xy
            color = PolygonStatus.VALIDE.color_gt if row["has_OCR"] else PolygonStatus.MANQUANT.color_gt
            fig.add_trace(go.Scatter(
                x=list(x), y=list(y),
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False
            ))


def add_ocr_trace(fig, df_ocr_img):
    for _, row in df_ocr_img.iterrows():
        if row["polygon_status"] == PolygonStatus.VALIDE:
            x, y = row["polygon"].exterior.xy
            color = PolygonStatus.MANQUANT.color_ocr if row["region_id_GT"] is None else PolygonStatus.VALIDE.color_ocr
            fig.add_trace(go.Scatter(
                x=list(x), y=list(y),
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False
            ))


# Dashboards
def show_img_gt_ocr(df_images, df_gt, df_ocr):

    st.markdown("<h4 style='margin-bottom:12px;'>Visualisation GT / OCR</h4>", unsafe_allow_html=True)

    # Ligne du haut
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 4])

    with col1:
        selected_image = st.selectbox("Image :", df_images["image_filename"].tolist())

    with col2:
        show_img = st.checkbox("Image", True)

    with col3:
        show_gt = st.checkbox("GT", True)

    with col4:
        show_ocr = st.checkbox("OCR", True)

    # Données filtrées
    df_gt_img = df_gt[df_gt["image_filename"] == selected_image]
    df_ocr_img = df_ocr[df_ocr["image_filename"] == selected_image]

    gt_linked = df_gt_img["has_OCR"].sum()
    gt_unlinked = len(df_gt_img) - gt_linked

    ocr_linked = df_ocr_img["region_id_GT"].notna().sum()
    ocr_unlinked = len(df_ocr_img) - ocr_linked

    # Légende (pas d’overlay)
    with col5:
        st.markdown(
            f"""
            <div style="font-size:13px; line-height:1.2;">
                🔵 GT liés : {gt_linked}<br>
                🔴 GT sans OCR : {gt_unlinked}<br>
                🟢 OCR liés : {ocr_linked}<br>
                🟡 OCR sans GT : {ocr_unlinked}
            </div>
            """,
            unsafe_allow_html=True
        )

    # Bouton agrandir / réduire
    fullscreen = st.toggle("Agrandir l’image")

    # Image
    img_b64 = df_images[df_images["image_filename"] == selected_image]["image_base64"].iloc[0]

    fig = add_image_trace(
        img_b64,
        fullscreen=fullscreen
    )

    if show_gt:
        add_gt_trace(fig, df_gt_img)

    if show_ocr:
        add_ocr_trace(fig, df_ocr_img)

    st.plotly_chart(fig, use_container_width=True)
    
def show_synthese_ocr(df_ocr):

    st.markdown("### Synthèse OCR")

    st.metric("Nombre total de mots OCR", len(df_ocr))
    st.metric("Confiance moyenne", round(df_ocr["confidence"].mean(), 3))
    st.metric("Confiance médiane", round(df_ocr["confidence"].median(), 3))

    st.write("#### Répartition des niveaux de confiance")
    st.bar_chart(df_ocr["confidence_level"].value_counts())
    
def show_stats_globales(df_ocr):

    st.markdown("### Statistiques globales OCR")

    # Histogramme des confiances
    fig = px.histogram(
        df_ocr,
        x="confidence",
        nbins=30,
        title="Histogramme des confiances OCR",
        color_discrete_sequence=["#4C78A8"]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Confiance moyenne par image
    df_stats = df_ocr.groupby("image_filename")["confidence"].mean().reset_index()
    st.write("#### Confiance moyenne par image")
    st.dataframe(df_stats)

    
def show_outliers_graph(df_ocr):

    st.markdown("### Outliers OCR — Visualisation graphique")

    # Sélecteur du seuil
    seuil = st.slider(
        "Seuil de confiance",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )

    # Filtrage des outliers
    outliers = df_ocr[df_ocr["confidence"] < seuil]

    st.write(f"Nombre d’outliers détectés : {len(outliers)}")

    if len(outliers) == 0:
        st.info("Aucun outlier pour ce seuil.")
        return

    # Graphique
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=outliers["image_filename"],
        y=outliers["confidence"],
        mode="markers",
        marker=dict(
            size=10,
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
        title=f"Outliers OCR — seuil={seuil}",
        xaxis_title="Image",
        yaxis_title="Confidence",
        xaxis_tickangle=45,
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)
    

def show_heatmap_confiance(df_ocr):

    st.markdown("### Heatmap de confiance OCR par image")

    df_stats = df_ocr.groupby("image_filename")["confidence"].mean().reset_index()

    fig = px.imshow(
        [df_stats["confidence"]],
        labels=dict(x="Image", color="Confiance moyenne"),
        x=df_stats["image_filename"],
        color_continuous_scale="RdYlGn"
    )

    fig.update_layout(height=300)

    st.plotly_chart(fig, use_container_width=True)

