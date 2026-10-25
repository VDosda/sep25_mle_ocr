from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale_image_features(df):
    # Colonnes de features images à scaler
    FEATURES = [
        "sharpness", "contrast", "brightness", "noise",
        "orientation", "mirror", "density", "uniformity",
    ]

    # Copie pour éviter de modifier df en place
    df_out = df.copy()

    # Extraction des features
    X = df_out[FEATURES].values

    # Scaling uniquement sur les features images
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Remise dans le DataFrame
    df_out[FEATURES] = X_scaled

    return df_out