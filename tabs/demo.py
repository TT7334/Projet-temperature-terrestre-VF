import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def demo():
    # image
    st.image("./assets/terrebrule.jpg", use_column_width=True)
    # Sous-titre "dataviz"
    st.markdown(f'<h2 style="text-align: center; text-decoration: underline;">Prédictions</h2>', unsafe_allow_html=True)
    st.write("Maintenant que nous avons cerné le modèle le plus performant pour faire des prédictions, essayons de l'appliquer à nos données et comparons avec les études déjà présentes")
    st.markdown('<hr style="border: 2px solid #D3D3D3">', unsafe_allow_html=True)
    
    st.write("Tout d'abord, si nous regardons les graphiques de prévisions, ils n'évoluent pas de manière la plus positive. En effet, que ce soit pour les températures ou les emisisons de CO2, ils prévoient des augmentations dans les années à venir:")
    
    st.image("./assets/courbe_prevision_co2.png", use_column_width=True)
    st.image("./assets/courbe_prevision_temp.png", use_column_width=True)

    
    st.write("Nous pouvons maintenant regarder, en tapant l'année que l'on souhaite, les prévisions de notre modèle:")

    # Chargement des données de prévision
    previsions_co2= pd.read_csv('./data/predictions_co2.csv', sep=",")
    previsions_temp = pd.read_csv('./data/predictions_temperature.csv', sep=",")
    # Convertir les colonnes de date en datetime si nécessaire
    previsions_temp['ds'] = pd.to_datetime(previsions_temp['ds'])
    previsions_co2['ds'] = pd.to_datetime(previsions_co2['ds'])

    # Obtention des années disponibles dans les données
    years = previsions_temp['ds'].dt.year.unique()
    years = [year for year in years if year >= 2024]
    # Widget de sélection d'année
    selected_year = st.selectbox("Sélectionnez une année pour les prévisions", years)

    # Filtrage des données en fonction de l'année sélectionnée
    temperature_forecast = previsions_temp[previsions_temp['ds'].dt.year == selected_year]
    co2_forecast = previsions_co2[previsions_co2['ds'].dt.year == selected_year]

    if not temperature_forecast.empty and not co2_forecast.empty:
        formatted_temp = "{:.2f}".format(temperature_forecast['yhat'].values[0])
        formatted_co2 = "{:.2f}".format(co2_forecast['yhat'].values[0])
        st.write(f"**Prévision de différence de température pour {selected_year}: {formatted_temp}**")
        st.write(f"**Prévision d'émission (en millions de tonnes) de CO2 pour {selected_year}: {formatted_co2}**")
    else:
        st.write("Aucune prévision disponible pour cette année.")

    st.write("Si nous nous arrêtons maintenant sur l'année qui nous intéressait initialement, 2050, nos modèles prévoient:")
    
    temp_2050 = 1.82 
    co2_2050 = 61108.63

    # Formatage des valeurs
    formatted_temp_2050 = "{:.2f}".format(temp_2050)
    formatted_co2_2050 = "{:.2f}".format(co2_2050)

    # Utilisation de st.metric pour afficher les valeurs
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Prévision de Température pour 2050", value=formatted_temp_2050)
    with col2:
        st.metric(label="Prévision d'Émission (en millions de tonnes) de CO2 pour 2050", value=formatted_co2_2050)

    st.write("Comme nous pouvons le constater, les prédictions ne sont pas **optimistes** puisque notre modèle prévoit pour 2050 une **différence de température de +1.82°** par rapport au référentiel et de **plus 61.108 gigatonnes de tonnes de CO2.**")
    st.write("Entant donné la forte corrélation entre les températures et les emissions de CO2, nottament les emissions provenant de nos modes de consommation, **si celles-ci ne baissent pas dans le temps, nous risquons d'avoir des augmentations de températures constantes.**")

    st.write("Pour s'assurer de la cohérences de notre modèle, nous avons comparés nos résultats avec les données du GIEC, experts reconnus sur la question.")
    st.write("Nos prédictions par rapport au GIEC sont plus optimistes comme nous le revèle ce tableau ci-dessous:")
    st.image("./assets/previsions GIEC 2050.png", use_column_width=True)
    st.write("Les lignes correpondent aux différents scénarios établis par le GIEC en matière d'émissions de Gaz à effet de serre.")
    st.write("La ligne qui nous intéresse est l'intermédiaire car ce scénario part du principe que nous gardons nos modes de consommations actuels, sans rien changer")
    st.write("On peut voir que sur la période 2041 - 2060, **leur prévision est de +2°**, ce qui est plus **pessimiste** que notre modèle, mais reste tout de même cohérent.")
    st.write("Ceci s'explique par le fait que Le Giec prend en compte toutes les émissions de gaz à effet de serre et pas seulement le CO2. Les variables exogènes doivent être aussi mieux prises en compte.")
