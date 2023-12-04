import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def introduction_projet():
#PARTIE "Introduction du projet"
    # Image de garde
    st.image("./assets/glacierfondu.jpg", use_column_width=True)

    # Titre de la page
    st.markdown(f'<h1 style="white-space: nowrap;"><u>Projet fil rouge: Température terrestre</u></h1>', unsafe_allow_html=True)

    # Sous-titre "Rapport final"
    st.markdown(f'<h2 style="text-align: center;">Rapport final</h2>', unsafe_allow_html=True)
    st.markdown('<hr style="border: 2px solid #D3D3D3">', unsafe_allow_html=True)

    # Introduction et contexte
    st.subheader("I. Contexte :")
    st.write("En premier lieu, il est important de parler du climat dans sa globalité. En effet, depuis le 21ième siècle, ce sujet est de plus en plus prégnant dans les conversations. Il y a une raison à cela, celui-ci se dégrade fortement. Nous commençons seulement à en ressentir les conséquences : saisons estivales de plus en plus chaudes, pluies torrentielles dans certaines régions de la terre, sécheresses intensives dans d’autres. Et cela va de mal en pire: Cette dégradation est due principalement à l’intervention humaine sur terre et son système productiviste. Quasiment tous les secteurs ont un impact sur notre environnement. C’est un sujet donc devenu naturellement systémique puisqu’il est impacté par presque tous les secteurs de nos sociétés modernes. Le problème majeur est aujourd’hui la méconnaissance des enjeux climatiques que cela représente. Selon un rapport des Nations Unies en avril 2022 (https://www.nationalgeographic.fr/environnement/2017/09/le-veritable-cout-du-changement-climatique), le dérèglement climatique représente un coût financier énorme. Sans parler des coûts sociaux, humains et géopolitiques qui vont s’intensifier. Que fera-t-on lorsque des migrations de population seront contraintes de quitter leurs pays à cause du réchauffement climatique. Comment nos sociétés vont-elles s’organiser ? Les défis sont majeurs et encore trop peu de personnes en ont conscience.")
    st.markdown('<hr style="border: none; border-top: 2px solid #D3D3D3; width: 50%;">', unsafe_allow_html=True)

    # Objectifs du projet
    st.subheader("II. Objectifs du projet :")
    st.write("Pour ce projet, nous allons nous concentrer sur le dérèglement climatique. Ce dérèglement se matérialise principalement par un réchauffement sur la surface de la terre, que ce soient les terres ou les airs qui la compose.Les objectifs sont donc, en se basant sur des dataset accessibles gratuitement, de démontrer deux choses principales: ")
    st.write("<strong>1. Objectif 1 : Constater le réchauffement climatique et l’augmentation des émissions de CO2 :</strong>", unsafe_allow_html=True)
    st.write("   Notre premier objectif consistera à faire le constat qu’il y a bien un réchauffement climatique et ce, dans tous les coins de la planète. Pour répondre à cet objectif, nous essayerons d’établir ce constat par différentes visualisations et corrélations que ce soit, au niveau planétaire, des hémisphères, d’un continent comme l’Europe et d’un pays comme la France. Également, il sera intéressant de voir si les émissions de gaz à effet de serre, notamment le CO2, jouent un rôle dans ce réchauffement.")
    st.write("<strong>2. Objectif 2 : Modéliser un algorithme de machine learning pour tenter de prédire les changements de températures pour les 50 prochaines année, à l’échelle mondiale :</strong>", unsafe_allow_html=True)
    st.write("   Notre variable cible, si l’on se réfère au premier dataset, sera la température des années à venir (nouvelle variables/colonnes dans notre tableau) ou l’algorithme essayera d’anticiper les anomalies de températures qui pourraient subvenir, en attendant les réelles mesures qui seront prises durant ces années")
    st.markdown('<hr style="border: none; border-top: 2px solid #D3D3D3; width: 50%;">', unsafe_allow_html=True)

    # Datasets utilisés
    st.subheader("Datasets utilisés :")
    st.write("Avant de regarder les datasets, nous aborderons plusieurs notions importantes que nous détaillons ci-dessous :")
    st.write("<strong>1. L’anomalie de température pour mesurer le réchauffement climatique :</strong>", unsafe_allow_html=True)
    st.write("Valeur relative, exprimée en degrés Celsius correspondant à l’écart, positif ou négatif, entre la température mesurée et la température moyenne de référence correspondante. Elle représente l'intervalle de confiance à 95 %. Ces références nous permettent de calculer les températures en valeurs absolues.")
    st.write("<strong>2. Le CO2 (ou dioxyde de carbone) :</strong>", unsafe_allow_html=True)
    st.write("Cette fois-ci, le dioxyde de carbone est mesuré de manière plus « normal », sans faire référence à un référentiel. Ce ne sont pas des anomalies. Grace à des capteurs, la quantité de CO2 dégagée dans l’atmosphère est mesurée, à intervalles réguliers de temps.")

    st.write("Concentrons-nous maintenant sur les datasets que nous avons utilisés pour mener à bien ce projet :")

    st.write("<strong>1. Dataset sur les anomalies de températures :</strong>", unsafe_allow_html=True)
    data_temp = pd.read_csv("./data/anomalie_temperature_globale.csv", sep=",", encoding='cp1252')
    pays_uniques = data_temp["Area"].unique()
    selected_pays = st.selectbox("Si vous le souhaitez, vous pouvez sélectionner un pays de votre choix :", pays_uniques)
    if selected_pays == "":
        st.write(data_temp)
    else:
        filtered_data = data_temp[data_temp["Area"] == selected_pays]
        st.write(filtered_data)
    st.write("Nous sommes en présence d’un tableur en deux dimensions composées de 9 656 lignes et 66 colonnes (variables).<br> En colonne sont répertoriées les années de mesures, commençant de 1961 et allant jusqu’à 2022.<br> Nous avons en colonnes d'autres informations comme les pays du monde, les mois de mesures et si la mesure est une anomalie de températures (« temperature change ») et l’écart-type par rapport au référenciel de comparaison (Standard Deviation). <br>Les colonnes « Code  » sont les codifications des différentes colonnes catégorielles: Le pays (Area Code), Les mois de mesure (Months Code) et les Eléments (Element code).<br> Nous avons également dans la variable « Area », en plus des pays du monde entier, des « zones » géographiques comme le monde (World), l’Europe, l’Asie… qui nous seront utiles pour notre partie de modélisation. ",unsafe_allow_html=True)

    st.write("<strong>2. Dataset sur les émissions de CO2 et gaz à effet de serre :</strong>", unsafe_allow_html=True)
    data_co2 = pd.read_csv("./data/co2_global_non_nettoye.csv", sep=",", encoding='cp1252')
    pays_uniques2 = data_co2["country"].unique()
    selected_pays2 = st.selectbox("Si vous le souhaitez, vous pouvez sélectionner un pays de votre choix :", pays_uniques2)
    if selected_pays2:
        filtered_data2 = data_co2[data_co2["country"] == selected_pays2]
    else:
        filtered_data2 = data_co2
    st.write(filtered_data2)
    st.write("Nous sommes en présence d’un tableur en deux dimensions composées de 50 598 lignes et 79 colonnes (variables).<br> En colonne sont répertoriées les mesures de CO2 au global (« CO2 »). Ce taux est ensuite mesuré avec d’autres éléments: le Co2 par habitant, le Co2 par Energies, Co2 par PIB… <br>Nous avons en colonnes d'autres informations comme les pays du monde, les années de mesures, les iso_code des pays, leurs PIB (gdp) ainsi que leur population à l’année de mesure.<br> Nous avons également dans la variable « country », en plus des pays du monde entier, des « zones » géographiques comme le monde (World), l’Europe, l’Asie… qui nous seront utiles pour notre partie de modélisation. ",unsafe_allow_html=True)

    st.markdown("Ces données sont disponibles librement sur le site de la Nasa et sur le un github orchestré par l’association « Our World In Data », voici les liens:")
    st.markdown("[Site de la Nasa](https://data.giss.nasa.gov/gistemp/)")
    st.markdown("[GitHub Our World In Data](https://github.com/owid/co2-data)")
