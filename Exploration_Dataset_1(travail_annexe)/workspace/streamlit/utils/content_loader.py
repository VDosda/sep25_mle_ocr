from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1] / "contents"

def load_md(filename: str) -> str:
    """Charge un fichier MD et renvoie son contenu."""
    path = BASE_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

