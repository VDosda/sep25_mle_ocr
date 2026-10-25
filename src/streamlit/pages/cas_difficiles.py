import streamlit as st

st.header("Arbitrer les cas difficiles")
st.markdown("""
- 1er extrait : 67 documents où le SVM hésite entre ses deux premières classes, 56,7 % de bonnes réponses
- Trois arbitres tranchent entre ces deux classes, les autres documents ne changent pas

| Arbitre | Corrigés | Cassés | Gain net | Accuracy globale |
|---|---|---|---|---|
| Aucun (SVM seul) | 0 | 0 | 0 | 86,9 % |
| LLM | 14 | 5 | +9 | 89,4 % |
| Position des mots | 11 | 17 | −6 | 85,3 % |
| Détection de tableau | 13 | 16 | −3 | 86,1 % |

- Seul le LLM aide, mais il est trop lent
- Extrait final : un méta-modèle sur les documents incertains en corrige 8, sans en casser, un gain fragile
""")
