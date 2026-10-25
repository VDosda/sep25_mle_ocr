import json
import os

# Pointer vers data/schema/
BASE_DIR = "/workspace/data/schema"

def load_schema(model):
    # Charger le schéma commun
    with open(os.path.join(BASE_DIR, "common_schema.json"), "r") as f:
        common = json.load(f)

    # Charger le schéma spécifique
    specific_path = os.path.join(BASE_DIR, f"schema_{model}.json")
    if os.path.exists(specific_path):
        with open(specific_path, "r") as f:
            specific = json.load(f)
    else:
        specific = {"extra_required": [], "extra_schema": {}}

    # Fusionner
    required = common["required_columns"] + specific["extra_required"]
    schema = {**common["schema"], **specific["extra_schema"]}

    return {
        "required_columns": required,
        "schema": schema
    }
