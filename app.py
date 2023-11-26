import streamlit as st

#PARTIE SIDEBAR
color_code="#4628DD"
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: ##D3D3D3;
    }
</style>
""", unsafe_allow_html=True)

#A récupérer le logo de datascientest
st.sidebar.image("./assets/logo_datascientest.png")

# Case à cocher pour la navigation entre les pages
from tabs.introduction import introduction_projet
from tabs.preprocessing import prepro
from tabs.datavisualisation import dataviz
from tabs.methodologie import methodo
from tabs.demo import demo
from tabs.conclusion import conclusion

PAGES = {
    "Introduction du projet": introduction_projet,
    "Pre-processing":prepro,
    "Datavisualisation": dataviz,
    "Méthodologie de la modélisation": methodo,
    "Démonstration des prédictions": demo,
    "Conclusion":conclusion,
}
st.sidebar.markdown("## Sommaire:")
selection = st.sidebar.radio("Sélectionnez une page :", list(PAGES.keys()))
page = PAGES[selection]
page()

# Liste des contributeurs
st.sidebar.markdown("## Contributeurs:")
st.sidebar.markdown("Yasmine Kouyaté")
st.sidebar.markdown("Jean-Michel Deblaise")
st.sidebar.markdown("Youcef Arim")
st.sidebar.markdown("Théo Delafontaine")

# Promotion
st.sidebar.markdown("## Promotion:")
st.sidebar.write("avr23_continu_da")