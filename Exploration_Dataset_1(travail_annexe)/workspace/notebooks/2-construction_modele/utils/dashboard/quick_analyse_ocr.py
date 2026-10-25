import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.common import b64_to_html_img

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def dashboard_quality_quantiles_images(df_output, q=10):
    df = df_output.copy()

    # Quantiles équilibrés
    df["quality_bin"] = pd.qcut(
        df["quality_score"],
        q=q,
        duplicates="drop"
    ).astype(str)

    # Conversion base64 → HTML image
    df["segment_img"] = df["segment_b64"].apply(
        lambda b64: f'<img src="data:image/png;base64,{b64}" width="120px">'
    )

    # Tableau interactif Plotly
    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[80, 80, 150, 80],
                header=dict(
                    values=["Quantile", "ID", "Image", "Score"],
                    fill_color="lightgrey",
                    align="left"
                ),
                cells=dict(
                    values=[
                        df["quality_bin"],
                        df["id"],
                        df["segment_img"],
                        df["quality_score"].round(3)
                    ],
                    align="left",
                    height=80
                )
            )
        ]
    )

    fig.update_layout(
        title=f"Segments par quantiles de qualité (q={q})",
        height=900
    )

    fig.show()

    # Histogramme des quantiles
    fig2 = px.histogram(
        df,
        x="quality_bin",
        color="segment_index",
        title=f"Distribution des quantiles de qualité (q={q})",
        category_orders={"quality_bin": sorted(df["quality_bin"].unique())}
    )

    fig2.update_layout(
        xaxis_title="Quantiles de score qualité",
        yaxis_title="Nombre de segments",
        bargap=0.1
    )

    fig2.show()

    
# Dashboard 1 : Distribution du score qualité
def dashboard_quality(df_output):
    df = df_output.copy()
    df["quality_score"] = df["quality"].apply(lambda x: x["quality_score"])

    fig = px.histogram(
        df,
        x="quality_score",
        nbins=40,
        title="Distribution du score qualité",
        color="segment_index"
    )
    fig.update_layout(bargap=0.1)
    fig.show()


# Dashboard 2 : Contraste vs Qualité
def dashboard_contrast_vs_quality(df_output):
    df = df_output.copy()
    df["quality_score"] = df["quality"].apply(lambda x: x["quality_score"])
    df["contrast"] = df["quality"].apply(lambda x: x["contrast"])

    fig = px.scatter(
        df,
        x="contrast",
        y="quality_score",
        color="segment_index",
        hover_data=["id", "segment_index"],
        title="Qualité vs Contraste"
    )
    fig.show()


# Dashboard 3 : Distribution des tailles de segments
def dashboard_size(df_output):
    df = df_output.copy()
    df["size"] = df["quality"].apply(lambda x: x["size"])

    fig = px.histogram(
        df,
        x="size",
        nbins=50,
        title="Distribution des tailles de segments"
    )
    fig.show()


# Dashboard 4 : Distribution du ratio largeur/hauteur
def dashboard_ratio(df_output):
    df = df_output.copy()
    df["ratio"] = df["quality"].apply(lambda x: x["ratio"])

    fig = px.histogram(
        df,
        x="ratio",
        nbins=40,
        title="Distribution du ratio largeur/hauteur"
    )
    fig.show()


# Dashboard 5 : Table interactive avec images
def dashboard_table_segments(df_output):
    df = df_output.copy()

    # Préparation des colonnes
    df["quality_score"] = df["quality"].apply(lambda x: x["quality_score"])
    df["contrast"] = df["quality"].apply(lambda x: x["contrast"])
    df["segment_img"] = df["segment_b64"].apply(b64_to_html_img)

    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[40, 40, 200, 80, 80],
                header=dict(
                    values=["ID", "Segment", "Image", "Qualité", "Contraste"],
                    fill_color="lightgrey",
                    align="left"
                ),
                cells=dict(
                    values=[
                        df["id"],
                        df["segment_index"],
                        df["segment_img"],
                        df["quality_score"].round(3),
                        df["contrast"].round(3)
                    ],
                    align="left",
                    height=60
                )
            )
        ]
    )

    fig.update_layout(title="Table interactive des segments")
    fig.show()


# Dashboard 6 : Clustering des segments
def dashboard_clusters(df_output, n_clusters=4):
    from sklearn.cluster import KMeans

    df = df_output.copy()
    df["quality_score"] = df["quality"].apply(lambda x: x["quality_score"])
    df["contrast"] = df["quality"].apply(lambda x: x["contrast"])

    # Extraction des features
    X = df["features"].apply(pd.Series).fillna(0)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(X)

    fig = px.scatter(
        df,
        x="contrast",
        y="quality_score",
        color="cluster",
        hover_data=["id", "segment_index"],
        title=f"Clusters de segments (k={n_clusters})"
    )
    fig.show()
