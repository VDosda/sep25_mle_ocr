import streamlit as st

import outils

st.header("Combiner les modèles : le Stacking")
st.markdown("""
Deux sources, trois modèles : le texte (SVM) et la mise en page, vue par nos 85 mesures (XGBoost)
et par le CNN. Une régression logistique combine leurs probabilités.
""")

predictions = outils.load_predictions()
st.write("Accuracy sur les", len(predictions), "documents de test :")
st.dataframe(outils.results_table(predictions))
st.markdown("**91,9 %**, soit 5,2 points de plus que le texte seul : 193 erreurs au lieu de 319.")

st.sidebar.divider()
modele = st.sidebar.selectbox("Matrice de confusion du modèle", outils.TOUS_LES_MODELES)
st.pyplot(outils.plot_confusion_matrix(predictions, modele), width=outils.LARGEUR_GRAPHIQUE)

st.markdown("""
### 4 constats
- Le texte est le signal le plus fort : 50,6 % pour la meilleure approche visuelle (HOG), 86,9 % pour le texte (SVM)
- La règle de décision compte autant que la représentation : mêmes vecteurs TF-IDF caractères, 80,6 % avec le
  cosinus, 86,9 % avec un SVM
- Combiner deux lectures du même texte (mots + caractères) n'apporte presque rien ; combiner le texte, l'image et
  le CNN : 86,7 % → 89,5 % → 91,9 %
- Le LLM n'est pas compétitif seul (73,3 %, plusieurs secondes par document), mais il est utile en arbitre
""")
