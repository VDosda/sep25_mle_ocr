def classify_confidence(c):
    if c < 0.5:
        return "Faible"
    elif c < 0.8:
        return "Moyenne"
    return "Forte"
