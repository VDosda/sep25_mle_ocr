import numpy as np
import matplotlib.pyplot as plt

def correlation_heatmap_features_quality(
    df, 
    features =["sharpness", "contrast", "brightness", "noise","orientation", "mirror", "density", "uniformity", "doc_ratio"],
    target="img_quality_score"
):
    # Sélection des colonnes
    cols = features + [target]
    corr = np.abs(df[cols].corr().values)

    n = len(cols)

    # Masque pour n'afficher que la diagonale supérieure
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Palette diverging
    cmap = plt.cm.coolwarm

    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # Affichage des cases
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                value = corr[i, j]
                color = cmap((value + 1) / 2)  # normalisation [-1,1] → [0,1]
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))

                # Annotation
                ax.text(j + 0.5, i + 0.5, f"{value:.2f}",
                        ha="center", va="center", fontsize=10, color="black")

    # Axes
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)

    # Highlight de la target
    for idx, label in enumerate(ax.get_xticklabels()):
        if label.get_text() == target:
            label.set_color("red")

    for idx, label in enumerate(ax.get_yticklabels()):
        if label.get_text() == target:
            label.set_color("red")

    # Limites
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.invert_yaxis()

    plt.title("Heatmap Matplotlib — diagonale supérieure (influence des features sur la qualité)")
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax)
    plt.show()
