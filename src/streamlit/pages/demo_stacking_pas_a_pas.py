import time

import pandas as pd
import streamlit as st

import outils

st.header("Démo : le Stacking pas à pas")
st.markdown("""
Des documents jamais vus passent un par un dans nos 3 modèles, puis dans le stacking.
En vert les bonnes réponses, en rouge les erreurs.
""")

st.sidebar.divider()
jeu = st.sidebar.selectbox("Documents de test", list(outils.JEUX_DE_TEST),
                           index=len(outils.JEUX_DE_TEST) - 1)
documents = outils.load_test_documents(outils.JEUX_DE_TEST[jeu])

probas = outils.predict_pipeline(documents)

if st.sidebar.checkbox("Seulement les documents où les 3 modèles ne sont pas d'accord"):
    predites = {m: probas[m].idxmax(axis=1) for m in outils.MODELES}
    desaccord = (predites["texte"] != predites["image"]) | (predites["texte"] != predites["cnn"])
    documents = documents[desaccord.to_numpy()]
n = st.sidebar.slider("Nombre de documents", 1, len(documents), len(documents))
pause = st.sidebar.slider("Pause entre deux documents (secondes)", 0.0, 5.0, 1.0, 0.5)

documents = documents.sample(n, random_state=0)
etapes = ["texte", "image", "cnn", "stacking"]
justes = {etape: 0 for etape in etapes}

with st.container(horizontal_alignment="center"):
    lancer = st.button("Lancer la démo")
numero = st.slider("Document affiché (pour revenir en arrière)", 1, n, 1)

zone_image = st.empty()
st.markdown("#### 1. Les trois modèles")
colonnes = st.columns(3)
zones_modeles = {m: colonne.empty() for m, colonne in zip(outils.MODELES, colonnes)}
st.markdown("#### 2. Le stacking")
colonne_choix, colonne_tableau = st.columns(2)
zone_choix, zone_tableau = colonne_choix.empty(), colonne_tableau.empty()

for i in range(n if lancer else numero):
    filepath = documents["filepath"].iloc[i]
    classe = documents["label"].iloc[i]

    tableau = pd.DataFrame({etape: probas[etape].loc[filepath] for etape in etapes})
    predictions = tableau.idxmax()
    for etape in etapes:
        if predictions[etape] == classe:
            justes[etape] += 1

    if i < numero - 1:
        continue
    if i > numero - 1:
        time.sleep(pause)


    with zone_image.container(horizontal_alignment="center"):
        st.image(filepath, width=300, caption=f"Document {i + 1} / {n}, classe réelle : {classe}")

    for m, nom in outils.MODELES.items():
        couleur = "green" if predictions[m] == classe else "red"
        zones_modeles[m].markdown(f"""
**{nom}**
### :{couleur}[{predictions[m]} ({tableau[m].max():.0%})]
{justes[m]} / {i + 1} justes
""")

    suivis = [m for m in outils.MODELES if predictions[m] == predictions["stacking"]]
    if len(suivis) == 3:
        choix = "Les 3 modèles sont d'accord : le stacking garde leur réponse."
    elif suivis:
        choix = f"Les modèles ne sont pas d'accord : le stacking suit **{' et '.join(suivis)}**."
    else:
        choix = "Le stacking choisit une classe qu'aucun des 3 modèles n'a mise en premier."
    couleur = "green" if predictions["stacking"] == classe else "red"
    zone_choix.markdown(f"""
**{outils.NOTRE_MODELE}**
### :{couleur}[{predictions['stacking']} ({tableau['stacking'].max():.0%})]
{choix}

**{justes['stacking']} / {i + 1} justes**
""")
    zone_tableau.dataframe(tableau.style.format("{:.0%}").background_gradient(cmap="Blues", vmin=0, vmax=1))

if lancer:
    st.markdown("Score final : " + ", ".join(f"{etape} {justes[etape]} / {n}" for etape in etapes))
