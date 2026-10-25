import time

import streamlit as st

import outils


def jauge(resultats, n):
    """Une case par document : verte si le modèle a juste, rouge s'il se trompe, grise s'il n'est pas encore passé."""
    couleurs = ["#21c354" if juste else "#ff4b4b" for juste in resultats] + ["#e6e6e6"] * (n - len(resultats))
    cases = "".join(f'<div style="flex: 1; background: {couleur}"></div>' for couleur in couleurs)
    espace = "2px" if n <= 60 else "0px"
    return f'<div style="display: flex; gap: {espace}; height: 24px; border-radius: 4px; overflow: hidden">{cases}</div>'


st.header("Démo : notre modèle face au DiT de Hugging Face")
st.markdown("""
Notre stacking face au DiT de Microsoft (`microsoft/dit-base-finetuned-rvlcdip`), entraîné sur RVL-CDIP.
""")

st.sidebar.divider()
jeu = st.sidebar.selectbox("Documents de test", list(outils.JEUX_DE_TEST),
                           index=len(outils.JEUX_DE_TEST) - 1)
documents = outils.load_test_documents(outils.JEUX_DE_TEST[jeu])
n = st.sidebar.slider("Nombre de documents", 1, len(documents), len(documents))
pause = st.sidebar.slider("Pause entre deux documents (secondes)", 0.0, 3.0, 0.05, 0.05)

documents = documents.sample(n, random_state=0)
modeles = [outils.NOTRE_MODELE, outils.MODELE_HF]
succes = {nom: 0 for nom in modeles}
erreurs = {nom: 0 for nom in modeles}
resultats = {nom: [] for nom in modeles}

predictions = {outils.NOTRE_MODELE: outils.predict_stacking(documents),
               outils.MODELE_HF: outils.predict_dit(list(documents["filepath"]))}

with st.container(horizontal_alignment="center"):
    lancer = st.button("Lancer la démo")
numero = st.slider("Document affiché (pour revenir en arrière)", 1, n, 1)

zone_image = st.empty()
colonnes = st.columns(2)
zones_scores = {nom: colonne.empty() for nom, colonne in zip(modeles, colonnes)}

for i in range(n if lancer else numero):
    classe = documents["label"].iloc[i]
    for nom in modeles:
        if predictions[nom][i] == classe:
            succes[nom] += 1
        else:
            erreurs[nom] += 1
        resultats[nom].append(predictions[nom][i] == classe)

    if i < numero - 1:
        continue
    if i > numero - 1:
        time.sleep(pause)

    # le document, au centre
    with zone_image.container(horizontal_alignment="center"):
        st.image(documents["filepath"].iloc[i], width=300,
                 caption=f"Document {i + 1} / {n}, classe réelle : {classe}")

    for nom in modeles:
        with zones_scores[nom].container():
            st.markdown(f"#### {nom}\nprédit : **{predictions[nom][i]}**")
            st.markdown(jauge(resultats[nom], n), unsafe_allow_html=True)
            st.markdown(f"### :green[✔ {succes[nom]} succès]\n### :red[✘ {erreurs[nom]} erreurs]")
