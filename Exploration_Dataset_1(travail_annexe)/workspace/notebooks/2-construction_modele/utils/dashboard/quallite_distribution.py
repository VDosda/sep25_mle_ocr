import numpy as np
import matplotlib.pyplot as plt

def quality_category_distribution(df, column="quality_category"):
    counts = df[column].value_counts()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index, counts.values, color=["green", "orange", "red"])
    plt.title("Répartition des images par catégorie de qualité")
    plt.xlabel("Catégorie")
    plt.ylabel("Nombre d'images")
    plt.grid(True, alpha=0.3)
    plt.show()
    
def compare_quality_category_distribution(df_before, df_after,step="preprocessing"):
    counts_before = df_before["quality_category"].value_counts().sort_index()
    counts_after = df_after["quality_category"].value_counts().sort_index()

    categories = counts_before.index
    x = range(len(categories))
    width = 0.35

    plt.figure(figsize=(12,6))
    plt.bar(x, counts_before.values, width=width, label="Avant", alpha=0.7)
    plt.bar([i + width for i in x], counts_after.values, width=width, label="Après", alpha=0.7)

    plt.xticks([i + width/2 for i in x], categories, rotation=45)
    plt.ylabel("Nombre d'images")
    plt.title(f"Répartition des images par catégorie — Avant vs Après {step}")
    plt.legend()
    plt.tight_layout()
    plt.show()
