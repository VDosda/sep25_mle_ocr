import streamlit as st

import outils

st.header("Preuve de concept : valider la méthode sur un cas simple")
st.markdown("Dataset 1 : 165 documents, 3 classes, 33 documents de test.")

mesures, cosinus = st.tabs(["6 mesures, 5 classifieurs", "Centroïdes + cosinus"])

with mesures:
    st.markdown("""
- 6 mesures par image, en niveaux de gris : largeur, hauteur, format, luminosité, contraste, part de pixels sombres
- 5 classifieurs du cours, réglés par validation croisée sur le train (GridSearchCV, RandomizedSearchCV)
- Le test (33 documents) ne sert qu'à la mesure finale
- Le SVM est premier, mais les 4 premiers ne sont séparés que par 2 documents

| Classifieur | Accuracy |
|---|---|
| SVM | 75,8 % |
| KNN | 69,7 % |
| XGBoost | 69,7 % |
| Random Forest | 69,7 % |
| Arbre de décision | 63,6 % |

Mêmes erreurs pour tous : aucun e-mail mal classé, des CV pris pour des e-mails ou des publications,
des publications prises pour des CV.
""")
    st.pyplot(outils.plot_confusion_matrices({
        "SVM\n75,8 %": [[8, 0, 0], [3, 9, 2], [0, 3, 8]],  # \n : passage à la ligne dans le titre
        "KNN, XGBoost, Random Forest\n69,7 %": [[8, 0, 0], [4, 8, 2], [0, 4, 7]],
        "Arbre de décision\n63,6 %": [[8, 0, 0], [4, 8, 2], [0, 6, 5]],
    }, ["email", "resume", "scientific_publication"]), width=outils.LARGEUR_FIGURE)

with cosinus:
    st.markdown("""
Un vecteur par document (pixels, HOG ou texte lu par l'OCR), puis :
1. un centroïde par classe : la moyenne des vecteurs de ses documents d'entraînement ;
2. un document de test prend la classe du centroïde le plus proche au sens du cosinus.

| Source | Représentation | Décision | Accuracy |
|---|---|---|---|
| Image | 6 mesures globales | SVM | 75,8 % |
| Image | Pixels bruts | Cosinus | 78,8 % |
| Image | HOG | Cosinus | 90,9 % |
| Texte (OCR) | TF-IDF caractères | Cosinus | 93,9 % |
| Texte (OCR) | TF-IDF mots, sans mots vides | Cosinus | **100 %** |
""")

st.markdown("""
- Toutes les méthodes battent le hasard (33 %)
- Le texte fait mieux que l'image
- 33 documents de test : une erreur coûte 3 points
- Jeu trop petit et trop propre
""")
