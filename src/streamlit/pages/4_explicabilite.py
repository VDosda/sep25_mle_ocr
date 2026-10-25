import streamlit as st

import outils

EXEMPLE = {"classe": "invoice", "seulement_les_erreurs": True, "numero": 15}

st.header("Explicabilité du CNN")
st.markdown("Grad-CAM : en rouge, les zones qui font monter le score d'une classe.")

par_classe, erreurs, en_direct = st.tabs(["Un document par classe", "Quand il se trompe", "Un document en direct"])

with par_classe:
    st.markdown("""
- Le CNN regarde la mise en page, pas le texte : l'en-tête d'un e-mail, les cases d'un formulaire, un tableau,
  une liste à puces, un graphique
""")
    st.image("src/streamlit/images/grad_cam_par_classe.png", width=outils.LARGEUR_FIGURE)

with erreurs:
    st.markdown("""
- Il confond surtout des documents qui se ressemblent vraiment : une facture et un formulaire
- Il est trop sûr de lui : 99 à 100 % de confiance, même quand il se trompe
""")
    st.image("src/streamlit/images/grad_cam_erreurs.png", width=outils.LARGEUR_GRAPHIQUE)

with en_direct:
    test = outils.load_predictions()
    st.sidebar.divider()
    classe = st.sidebar.selectbox("Classe réelle du document", outils.CLASSES,
                                  index=outils.CLASSES.index(EXEMPLE["classe"]))
    documents = test[test["label"] == classe]
    if st.sidebar.checkbox("Seulement les documents où le CNN se trompe", value=EXEMPLE["seulement_les_erreurs"]):
        documents = documents[documents[outils.MODELES["cnn"]] != classe]
    numero = st.sidebar.slider("Numéro du document", 1, len(documents), min(EXEMPLE["numero"], len(documents)))
    filepath = documents["filepath"].iloc[numero - 1]

    probas = outils.predict_cnn([filepath])[0]  # vraie prédiction
    prediction = outils.CLASSES[probas.argmax()]
    st.markdown(f"Classe réelle : **{classe}**. Le CNN prédit **{prediction}** ({probas.max():.0%}).")

    if prediction == classe:
        st.pyplot(outils.plot_grad_cam(filepath, [prediction]), width=outils.LARGEUR_FIGURE)
    else:  # le CNN se trompe
        st.pyplot(outils.plot_grad_cam(filepath, [prediction, classe]), width=outils.LARGEUR_FIGURE)
