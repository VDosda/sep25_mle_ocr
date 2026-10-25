import pandas as pd
import numpy as np
import json

# NORMALISATION DES DATAFRAMES 
def normalize_output(df, step_name, step_version):
    df["step_name"] = step_name
    df["step_version"] = step_version

    # Base64 doit être une string UTF-8
    df["image_base64"] = df["image_base64"].astype(str)

    validate_df(df, model=step_version)
    
    return df

# VALIDATION
from utils.schemas import load_schema

def validate_df(df, model):
    schema = load_schema(model)

    required = schema["required_columns"]
    types = schema["schema"]

    errors = []

    # Vérification des colonnes obligatoires
    for col in required:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    # Vérification des types
    for col, expected_type in types.items():
        if col not in df.columns:
            continue

        dtype = df[col].dtype

        if expected_type == "int" and not np.issubdtype(dtype, np.integer):
            errors.append(f"Column {col} must be int dtype")

        if expected_type == "float" and not np.issubdtype(dtype, np.floating):
            errors.append(f"Column {col} must be float dtype")

        if expected_type == "string" and dtype != object:
            errors.append(f"Column {col} must be string dtype")

    if errors:
        raise ValueError("\n".join(errors))

    print(f"DataFrame validated successfully for model '{model}'.")
    return True

