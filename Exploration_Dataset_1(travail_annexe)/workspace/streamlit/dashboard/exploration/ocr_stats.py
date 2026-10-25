import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import math

def show_confiance_ocr(df_ocr):

    st.markdown("### Analyse de la confiance OCR")

    page_size = 20
    images = sorted(df_ocr["image_filename"].unique())
    num_pages = math.ceil(len(images) / page_size)

    page = st.number_input(
        "Page",
        min_value=0,
        max_value=num_pages - 1,
        value=0,
        step=1,
        label="Page"
    )

    seuil = st.slider(
        "Seuil de confiance",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        key="seuil_confiance_ocr"
    )

    start = page * page_size
    end = start + page_size
    imgs = images[start:end]
    df_page = df_ocr[df_ocr["image_filename"].isin(imgs)]

    fig = go.Figure()

    fig.add_trace(go.Box(
        x=df_page["image_filename"],
        y=df_page["confidence"],
        boxpoints=False,
        name="Confidence",
        marker_color="lightgray"
    ))

    outliers = df_page[df_page["confidence"] < seuil]
    non_outliers = df_page[df_page["confidence"] >= seuil]

    fig.add_trace(go.Scatter(
        x=non_outliers["image_filename"],
        y=non_outliers["confidence"],
        mode="markers",
        marker=dict(size=6, color="lightgray"),
        hoverinfo="skip",
        name="Non-outliers"
    ))

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

    st.plotly_chart(fig, use_container_width=True)


def show_synthese_ocr(df_ocr):

    st.markdown("### Synthèse OCR")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nombre total de mots OCR", len(df_ocr))
    with col2:
        st.metric("Confiance moyenne", round(df_ocr["confidence"].mean(), 3))
    with col3:
        st.metric("Confiance médiane", round(df_ocr["confidence"].median(), 3))

    if "confidence_level" in df_ocr.columns:
        counts = df_ocr["confidence_level"].value_counts().reset_index()
        counts.columns = ["niveau", "count"]
        fig = px.bar(
            counts,
            x="niveau",
            y="count",
            title="Répartition des niveaux de confiance",
            color="niveau",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)


def show_stats_globales(df_ocr):

    st.markdown("### Statistiques globales OCR")

    fig_hist = px.histogram(
        df_ocr,
        x="confidence",
        nbins=30,
        title="Histogramme des confiances OCR",
        color_discrete_sequence=["#4C78A8"]
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    df_stats = df_ocr.groupby("image_filename")["confidence"].mean().reset_index()
    fig_mean = px.bar(
        df_stats,
        x="image_filename",
        y="confidence",
        title="Confiance moyenne par image",
        color="confidence",
        color_continuous_scale="RdYlGn"
    )
    fig_mean.update_layout(xaxis_tickangle=45, height=500)
    st.plotly_chart(fig_mean, use_container_width=True)


def show_export_outliers(df_ocr):

    st.markdown("### Outliers OCR (liste + export)")

    seuil = st.slider(
        "Seuil de confiance",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        key="seuil_export_outliers"
    )

    outliers = df_ocr[df_ocr["confidence"] < seuil]

    st.write(f"Nombre d’outliers : {len(outliers)}")

    fig = px.scatter(
        outliers,
        x="image_filename",
        y="confidence",
        color="confidence",
        color_continuous_scale="RdYlGn",
        hover_data=["text", "word_id"],
        title=f"Outliers OCR — seuil={seuil}",
    )
    fig.update_layout(xaxis_tickangle=45, height=600)
    st.plotly_chart(fig, use_container_width=True)

    csv = outliers.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger les outliers (CSV)",
        csv,
        "outliers.csv",
        "text/csv"
    )

def show_heatmap_confiance(df_ocr):

    st.markdown("### Heatmap / Dashboard interactif de confiance OCR")

    # Slider seuil
    seuil = st.slider(
        "Seuil de confiance (ligne rouge)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        key="heatmap_seuil"
    )

    # Calcul des confiances moyennes
    df_stats = df_ocr.groupby("image_filename")["confidence"].mean().reset_index()

    # Tri automatique du plus faible au plus fort
    df_stats = df_stats.sort_values("confidence").reset_index(drop=True)
    df_stats["rank"] = df_stats.index + 1

    # graphique en barres
    fig = px.bar(
        df_stats,
        x="image_filename",
        y="confidence",
        color="confidence",
        color_continuous_scale="RdYlGn",
        hover_data={
            "image_filename": True,
            "confidence": ":.3f",
            "rank": True,
            "delta_seuil": df_stats["confidence"] - seuil
        },
        title="Confiance moyenne par image (triée)"
    )

    # Ligne rouge = seuil visuel
    fig.add_hline(
        y=seuil,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Seuil = {seuil}",
        annotation_position="top left"
    )

    # Mise en forme
    fig.update_layout(
        height=600,
        xaxis_tickangle=45,
        coloraxis_colorbar=dict(
            title="Confiance",
            thickness=25,
            len=0.7,
            x=0.5,
            xanchor="center",
            orientation="h",
            tickfont=dict(size=14),
            titlefont=dict(size=16)
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def show_outliers_graph(df_ocr):

    st.markdown("### Outliers OCR — Visualisation graphique")

    seuil = st.slider(
        "Seuil de confiance (graphique outliers)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        key="seuil_outliers_graph"
    )

    outliers = df_ocr[df_ocr["confidence"] < seuil]

    st.write(f"Nombre d’outliers détectés : {len(outliers)}")

    if len(outliers) == 0:
        st.info("Aucun outlier pour ce seuil.")
        return

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
