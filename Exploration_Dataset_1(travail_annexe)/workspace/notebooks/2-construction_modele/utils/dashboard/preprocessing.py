import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

import cv2
from utils.common import b64_to_cv


# SHOW DASHBOARD (une seule image)

def show_dashboard(row):
    """Affiche une image avant et après prétraitement, avec ses features et son score de qualité."""

    img_before = b64_to_cv(row["image_base64_before"])
    img_after = b64_to_cv(row["image_base64"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # INPUT
    axes[0].imshow(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Image AVANT traitement (input)")
    axes[0].axis("off")

    # OUTPUT
    axes[1].imshow(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Image APRÈS traitement (output)")
    axes[1].axis("off")

    plt.suptitle(f"Comparaison Avant / Après — {row['image_filename']}", fontsize=16)
    plt.show()

    print("\n FEATURES")
    for k in ["sharpness", "contrast", "brightness", "noise",
              "orientation", "mirror", "density", "uniformity", "doc_ratio"]:
        print(f"{k:12s} : {row[k]:.3f}")

    print(f"\n Quality Score : {row['img_quality_score']:.3f}")


# DASHBOARD NOTEBOOK (slider interactif)

def dashboard_notebook_preprocessing(df):
    """Slider pour parcourir les images avant et après prétraitement."""

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(df)-1,
        step=1,
        description="Image index:",
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )

    out = widgets.Output()

    def update(change):
        with out:
            out.clear_output()
            idx = change["new"]
            show_dashboard(df.iloc[idx])

    slider.observe(update, names="value")

    display(slider)
    display(out)

    # Affiche la première image
    with out:
        show_dashboard(df.iloc[0])

        
        