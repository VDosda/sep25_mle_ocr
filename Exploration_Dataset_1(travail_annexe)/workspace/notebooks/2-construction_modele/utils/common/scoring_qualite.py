from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

def compute_quality_score(df, n_components=3):
    FEATURES = [
        "sharpness", "contrast", "brightness", "noise",
        "orientation", "mirror", "density", "uniformity", "doc_ratio"
    ]

    # Nettoyage doc_ratio
    col = df["doc_ratio"]

    # Cas 1 : toute la colonne est NaN
    if col.isna().all():
        df["doc_ratio"] = 1.0

    # Cas 2 : toutes les valeurs sont 0
    elif (col.min() == 0) and (col.max() == 0):
        df["doc_ratio"] = 1.0
        
    X = df[FEATURES].values

    # PCA multi-composantes
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X)

    # Variance expliquée (poids)
    weights = pca.explained_variance_ratio_

    # Score pondéré
    score = (components * weights).sum(axis=1)

    # Normalisation dans [0,1]
    score_norm = (score - score.min()) / (score.max() - score.min())

    df["img_quality_score"] = score_norm
    
    return df


def quality_kmeans_category(df, column="img_quality_score", k=3):
    values = df[column].dropna().values.reshape(-1, 1)
    km = KMeans(n_clusters=k, random_state=0).fit(values)
    labels = km.labels_

    # On mappe les labels à faible/moyenne/bonne selon l’ordre des centres
    centers = km.cluster_centers_.flatten()
    order = np.argsort(centers)

    mapping = {order[0]: "faible", order[1]: "moyenne", order[2]: "bonne"}

    df["quality_category"] = [mapping[l] for l in labels]
    return df
