import matplotlib.pyplot as plt
import cv2
import streamlit as st
import ipywidgets as widgets
from IPython.display import display
import pandas as pd
import numpy as np

from utils.common import b64_to_cv

def show(idx, df_input, df_output):
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2

    FEATURES = [
        "sharpness","contrast","brightness","noise",
        "orientation","mirror","density","uniformity","doc_ratio"
    ]

    row_in  = df_input.iloc[idx]
    row_out = df_output.iloc[idx]

    # Images
    img_in  = b64_to_cv(row_in["image_base64"])
    img_out = b64_to_cv(row_out["image_base64"])

    # Features AVANT
    feats_in = {k: row_in[k] for k in FEATURES}
    quality_in = row_in["img_quality_score"]

    # Features APRÈS
    feats_out = {k: row_out[k] for k in FEATURES}
    quality_out = row_out["img_quality_score"]

    # Delta
    delta = {k: feats_out[k] - feats_in[k] for k in FEATURES}

    # FIGURE
    fig = plt.figure(figsize=(32, 16))
    gs = fig.add_gridspec(2, 3)

    # Image originale
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB))
    ax1.set_title("Image ORIGINALE")
    ax1.axis("off")

    # Image préprocessée
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    ax2.set_title("Image PRÉPROCESSÉE")
    ax2.axis("off")

    # Texte comparatif
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    text = " COMPARAISON FEATURES\n\n"
    for k in FEATURES:
        text += f"{k:12s} : AVANT={feats_in[k]:.3f} | APRÈS={feats_out[k]:.3f} | Δ={delta[k]:+.3f}\n"

    text += f"\n Qualité AVANT : {quality_in:.3f}"
    text += f"\n Qualité APRÈS : {quality_out:.3f}"
    text += f"\n\n Fichier : {row_in['image_filename']}"
    ax3.text(0.05, 0.05, text, fontsize=12, family="monospace")

    # NORMALISATION GLOBALE SUR TOUT LE DATASET
    df_all = pd.concat([df_input[FEATURES], df_output[FEATURES]], axis=0)

    global_min = df_all.min().to_dict()
    global_max = df_all.max().to_dict()


    # Normalisation AVANT/APRÈS
    values_in_norm = []
    values_out_norm = []

    for k in FEATURES:
        vmin = global_min[k]
        vmax = global_max[k]

        if vmax == vmin:
            values_in_norm.append(0.5)
            values_out_norm.append(0.5)
        else:
            values_in_norm.append((feats_in[k] - vmin) / (vmax - vmin))
            values_out_norm.append((feats_out[k] - vmin) / (vmax - vmin))

    # RADAR AVANT / APRÈS
    ax4 = fig.add_subplot(gs[1, 0], polar=True)

    labels_loop = FEATURES + [FEATURES[0]]
    values_in_loop = values_in_norm + [values_in_norm[0]]
    values_out_loop = values_out_norm + [values_out_norm[0]]

    angles = np.linspace(0, 2 * np.pi, len(labels_loop))

    ax4.plot(angles, values_in_loop, label="AVANT (norm. global)", linewidth=2)
    ax4.fill(angles, values_in_loop, alpha=0.2)

    ax4.plot(angles, values_out_loop, label="APRÈS (norm. global)", linewidth=2)
    ax4.fill(angles, values_out_loop, alpha=0.2)

    ax4.set_xticks(angles)
    ax4.set_xticklabels(labels_loop)
    ax4.set_title("Radar — AVANT vs APRÈS (normalisation globale)")
    ax4.legend(loc="upper right")

    # RADAR DES DELTAS (normalisé globalement)
    ax5 = fig.add_subplot(gs[1, 1], polar=True)

    delta_values = [delta[k] for k in FEATURES]

    dmin = min(delta_values)
    dmax = max(delta_values)

    delta_norm = []
    for d in delta_values:
        if dmax == dmin:
            delta_norm.append(0.5)
        else:
            delta_norm.append((d - dmin) / (dmax - dmin))

    delta_loop = delta_norm + [delta_norm[0]]

    ax5.plot(angles, delta_loop, label="Δ normalisé global", linewidth=2, color="darkred")
    ax5.fill(angles, delta_loop, alpha=0.2, color="red")

    ax5.set_xticks(angles)
    ax5.set_xticklabels(labels_loop)
    ax5.set_title("Radar — Deltas uniquement (normalisation globale)")
    ax5.legend(loc="upper right")

    plt.suptitle(f"Comparaison AVANT / APRÈS — {row_in['image_filename']}", fontsize=20)
    plt.show()

# DASHBOARD NOTEBOOK (JUPYTER)

def dashboard_preprocessing_notebook(df_input, df_output):

    slider = widgets.IntSlider(
        value=0, min=0, max=len(df_input)-1, step=1,
        description="Index image", continuous_update=False
    )

    out = widgets.Output()

    def update(change):
        with out:
            out.clear_output()
            show(change["new"], df_input, df_output)

    slider.observe(update, names="value")

    display(slider)
    display(out)

    with out:
        show(0, df_input, df_output)


