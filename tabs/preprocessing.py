import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def prepro():
    # Image de garde
    st.image("./assets/preprocessing.jpg", use_column_width=True)

    # Sous-titre "Pre-processing"
    st.markdown(f'<h2 style="text-align: center; text-decoration: underline;">Pre-processing des données</h2>', unsafe_allow_html=True)
    st.write("Ce traitement préliminaire de nos données brutes a pour but d’arriver à des données exploitables et qui nous donneront un modèle plus performant. Nous allons explorer les étapes les plus importantes de ce preprocessing")
    st.markdown('<hr style="border: 2px solid #D3D3D3">', unsafe_allow_html=True)

    # Concernant le dataset des anomalies de températures
    st.subheader("I. Dataset sur les anomalies de températures :")
    st.write("Tout d’abord, comme évoqué précédemment, nous avons une colonne «Element» nous expliquant que dans ce dataset il y a deux « mesures » au même moment qui sont effectuées:") 
    st.write("La première est l’anomalie réelle (Temperature Change), puis par rapport à cette anomalie l’écart-type (standard déviation) est calculé. Cet écart-type ne nous intéresse pas et il multiplie par deux le dataset. D’autant plus que les chiffres sont souvent les mêmes.") 
    st.write("Par conséquent, nous décidons de supprimer les lignes avec pour modalités dans la variable Element: Standard Deviation.")
    data_temp = pd.read_csv("./data/anomalie_temperature_globale.csv", sep=",", encoding='cp1252')
    data_temp=data_temp[data_temp['Element'] != 'Standard Deviation']
    st.write(data_temp)
    st.write("Ensuite, nous avons des variables «Area Code», «Months code» et «Element Code» qui sont finalement une codification de variables déjà présentes dans le dataset. Nous décidons de les supprimer également car nous avons déjà l’information sous une autre forme.")
    st.write("La colonne «Unit» n’apporte aussi aucune information puisque la mesure en degré Celsius est la valeur unique de la colonne. Nous pouvons donc la supprimer.")
    st.write("Enfin, comme nous avons supprimé les lignes ou la modalité «Standard Devitation» était renseignée, il ne reste que la modalité «Temperature Change» dans la variable «Element»")
    columns_to_remove = ["Area Code", "Months Code", "Element Code", "Unit", "Element"]
    data_temp = data_temp.drop(columns=columns_to_remove)
    st.write(data_temp)
    st.markdown("Ensuite,nous avons **certaines variables qui sont concaténés par zone géographique**, en plus des pays (world, Europe, Asie …). Ces variables vont nous êtres très utiles pour notre visualisation. Pour la datavisualisation, nous décidons de tout garder car c’est primordial pour le type de visualisation que nous souhaitons réaliser (map intéractive).")
    st.markdown("En ce qui concerne les **valeurs manquantes (Nan)**, ce sont des mesures qui n’ont pas pu être prises lors de certaines années. Ceci s’expliquant par une antériorité trop importante ou une situation politique (guerre) trop tendu dans certains pays, empêchant la prise de mesure. Pour ce qui est de la partie datavisualisation, nous décidons de les laisser telles quelles car elles n’impactent pas notre visualisation.")
    st.write("En revanche, pour la **partie modélisation**, voici le traitement que nous avons choisi: (nous opérerons ces changements lors de la partie prédiction pour ne pas impacter nos visualisations)")
    st.write("- Premièrement, nous avons créé une **fonction qui a pour but de repérer les lignes composées d’au moins 60% de NaN.** Une fois repéré, la fonction va s’occuper de supprimer ces lignes. En effet, elles ne sont pas pertinentes pour notre modèle, n’apportent rien et risque de biaisé notre modèle.") 
    st.write("- Comme nos deux dataset ne commencent pas par les mêmes années, nous avons décidé de les rendre **cohérents et de les faire démarrer à la même année, à savoir 1961.**")
    st.write("- Pour les valeurs manquantes restantes après ces deux étapes de nettoyage, nous avons fait le choix de les remplacer par leurs **KNN plus proches voisins**. C'est un algorithme qui va choisir les K plus proches voisins de la valeur manquante et en faire la moyenne. Nous pensons que c’est pertinent car les anomalies de températures d’une année à l’autre restent des anomalies. C’est-à-dire que la différence reste faible (cf distribution des années), entre -1 et 1. Il est donc pertinent d’appliquer cette méthode pour ne pas qu’il y ait trop de conséquences négatives sur notre modèle.")
    st.markdown('<hr style="border: none; border-top: 2px solid #D3D3D3; width: 50%;">', unsafe_allow_html=True)
    
    # Concernant le dataset sur le CO2
    st.subheader("II. Dataset sur les émissions de CO2 et Gaz à effet de serre :")
    data_co2 = pd.read_csv("./data/co2_global_non_nettoye.csv", sep=",", encoding='cp1252')
    st.write(data_co2)
    st.write("Tout d’abord, comme nous l’avons constaté précédemment, ce dataset possède énormément de colonnes. Trop de colonnes. Toutes ne sont pas utiles pour notre datavisualisation ou notre modèle prédictif.")
    st.write("Nous avons donc fait le choix de supprimer des colonnes pour ne garder celles que nous jugeons le plus utiles, à savoir:")
    st.write("- **«co2»**: total des émissions de CO2") 
    st.write("- **«cement_co2»**: total CO2 émissions provenant du ciment")
    st.write("- **«coal_co2»**: total CO2 émissions provenant du charbon")
    st.write("- **«flaring_co2»**: total CO2 émissions provenant du torchage raffinerie")
    st.write("- **«gas_co2»**: Total Co2 émissions provenant du gaz")
    st.write("- **«oil_co2»**: CO2 émissions provenant du pétrole")
    st.write("- **«population»**: population de chaque pays")
    st.write("- **«gdp»**: PIB de chaque pays")
    st.write("- **«country»**: Pays")
    st.write("- **«year»**: Années")
    selected_columns = ['country', 'year','co2', 'cement_co2', 'coal_co2', 'flaring_co2', 'gas_co2', 'oil_co2', 'population', 'gdp']
    data_co2 = data_co2.loc[:, selected_columns]
    st.write(data_co2)
    st.write("Pour le reste, nous **appliquerons le même pre-processing que le dataset précédent pour garder de la cohérence**.")
    st.write("Concernant la partie modélisation, **nous ne garderons que la variable «co2»**, pour ensuite injecter ses prédictions dans les données des températures. Le reste va essentiellement nous servir pour la partie visualitation")
