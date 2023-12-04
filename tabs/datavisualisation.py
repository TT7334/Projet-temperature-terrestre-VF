import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, Legend, LegendItem, Range1d, FixedTicker, Label, LabelSet
from bokeh.layouts import gridplot, column, row
from bokeh.palettes import Category20
from bokeh.models.annotations import Title
from bokeh.models.tickers import FixedTicker
from bokeh.models import GeoJSONDataSource
from scipy.stats import gaussian_kde
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as py

def dataviz():
    # Concernant les anomalies de températures
    st.image("./assets/dataviz.jpg", use_column_width=True)
    # Sous-titre "dataviz"
    st.markdown(f'<h2 style="text-align: center; text-decoration: underline;">Datavisualisation</h2>', unsafe_allow_html=True)
    st.write("Ces Visualisations produites vont nous aider à bien comprendre ce qu'il en ressort et le constat que nous pourrons faire en se basant sur des chiffres, présentés de manière visuelle pour une meilleure compréhension.")
    st.markdown('<hr style="border: 2px solid #D3D3D3">', unsafe_allow_html=True)
    st.subheader("I. Concernant les anomalies de températures :")
    st.write("Pour rappel, un de nos objectifs est de constater le réchauffement climatique dans les différentes zones géographiques de la planète. Pour répondre à cet objectif, nous pouvons nous poser les questions suivantes et essayer d’y répondre par des visualisations :")
    st.write("**Est-ce qu’il y a eu une augmentation des températures à l’échelle globale à travers le temps ? A partir de quand ?**")
    data_temp_average = pd.read_csv('./data/moyenne anomalie temperature par hemisphere.csv', sep=",")
    source = ColumnDataSource(data=dict(
    x=data_temp_average['Year'],
    y1=data_temp_average['Glob'],
    y2=data_temp_average['NHem'],
    y3=data_temp_average['SHem']))

    def get_trendline(x, y):
        coefficients = np.polyfit(x, y, 1)
        polynomial = np.poly1d(coefficients)
        ys = polynomial(x)
        return ys

    trendline_global = get_trendline(data_temp_average['Year'], data_temp_average['Glob'])
    trendline_nh = get_trendline(data_temp_average['Year'], data_temp_average['NHem'])
    trendline_sh = get_trendline(data_temp_average['Year'], data_temp_average['SHem'])

    source.data['trend1'] = trendline_global
    source.data['trend2'] = trendline_nh
    source.data['trend3'] = trendline_sh

    p = figure(width=800, height=600, title="Anomalies des Températures Globales et par Hémisphères")
    p.line('x', 'y1', source=source, color="firebrick", alpha=0.6, legend_label="Global")
    p.line('x', 'trend1', source=source, color="firebrick", alpha=0.3, line_dash='dashed', legend_label="Tendance Global")
    p.line('x', 'y2', source=source, color="navy", alpha=0.6, legend_label="Hémisphère Nord")
    p.line('x', 'trend2', source=source, color="navy", alpha=0.3, line_dash='dashed', legend_label="Tendance H Nord")
    p.line('x', 'y3', source=source, color="green", alpha=0.6, legend_label="Hémisphère Sud")
    p.line('x', 'trend3', source=source, color="green", alpha=0.3, line_dash='dashed', legend_label="Tendance H Sud")

    hover = HoverTool(
        tooltips=[
        ("Année", "@x"),
        ("Temp_Global", "@y1{0.00}"),
        ("Temp_North", "@y2{0.00}"),
        ("Temp_South", "@y3{0.00}"),
        ("Tendance_Global", "@trend1{0.00}"),
        ("Tendance_North", "@trend2{0.00}"),
        ("Tendance_South", "@trend3{0.00}")],
        formatters={
            "@x": "numeral",
            "@y1": "numeral",
            "@y2": "numeral",
            "@y3": "numeral",
            "@trend1": "numeral",
            "@trend2": "numeral",
            "@trend3": "numeral"
        },
    )
    p.add_tools(hover)
    p.legend.location = "bottom_right"
    st.bokeh_chart(p)
    st.write("Grâce à ce graphique (construit à l’aide de la librairie Bokeh), nous avons croisé la variable année avec les mesures prises. Nous pouvons clairement constater une nette augmentation des anomalies de températures au fil des années.") 
    st.write("Cette tendance se confirme également pour les deux hémisphères de manière équivalente. En effet, les droites de régression, en pointillé, nous le montre.") 
    st.write("Nous notons toutefois que c’est l’hémisphère Nord qui témoigne de l’augmentation la plus marquée, surtout ces dernières décennies.") 
    st.write("Entre les années 1880 et 1920, les anomalies de températures sont stables et restent même négatives. A partir de 1940, nous voyons que les anomalies se rapprochent de zéro. Et ce n’est qu’à partir de 1980 ou toutes les mesures d’anomalies sont positives, montrant donc un réchauffement des températures. Ceci s’explique par l’industrialisation et la mondialisation de notre monde pour répondre à une demande toujours plus croissante. Nous produisons plus, nous exportons plus et ainsi de suite…") 
    
    st.write("**Est-ce que cette augmentation des anomalies se vérifie en Europe et en France ?**")
    data_temp = pd.read_csv("./data/anomalie_temperature_globale.csv", sep=",", encoding='cp1252')
    df_env_long = data_temp.melt(id_vars=['Area Code', 'Area', 'Months Code', 'Months', 'Element Code', 'Element', 'Unit'],var_name='Year', value_name='Temperature change')
    df_env_long['Year'] = df_env_long['Year'].str[1:].astype(int)
    data_temp_yearly = df_env_long.groupby(['Year', 'Area'], as_index=False)['Temperature change'].mean()
    eu_countries = ['Germany', 'Austria', 'Belgium', 'Bulgaria', 'Cyprus', 'Croatia', 'Denmark', 'Spain', 'Estonia','Finland', 'France', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg','Malta', 'Netherlands', 'Poland', 'Portugal', 'Czechia', 'Romania', 'Slovakia', 'Slovenia', 'Sweden','United Kingdom']
    eu_data = data_temp_yearly[data_temp_yearly['Area'].isin(eu_countries)]
    global_temp_change = data_temp_yearly.groupby('Year')['Temperature change'].mean().reset_index()
    europe_temp_change = eu_data.groupby('Year')['Temperature change'].mean().reset_index()
    france_temp_change = data_temp_yearly[data_temp_yearly['Area'] == 'France'].copy()

    trendline_global = np.polyfit(global_temp_change['Year'], global_temp_change['Temperature change'], 1)
    trendline_europe = np.polyfit(europe_temp_change['Year'], europe_temp_change['Temperature change'], 1)
    trendline_france = np.polyfit(france_temp_change['Year'], france_temp_change['Temperature change'], 1)

    source = ColumnDataSource(data={
        'x': global_temp_change['Year'],
        'y1': global_temp_change['Temperature change'],
        'y2': europe_temp_change['Temperature change'],
        'y3': france_temp_change['Temperature change'],
        'trend1': trendline_global[0] * global_temp_change['Year'] + trendline_global[1],
        'trend2': trendline_europe[0] * europe_temp_change['Year'] + trendline_europe[1],
        'trend3': trendline_france[0] * france_temp_change['Year'] + trendline_france[1],
    })
    p = figure(width=800, height=600, title="Anomalies des Températures Globales, en Europe et en France")
    p.line('x', 'y1', source=source, color="firebrick", alpha=0.6, legend_label="Global")
    p.line('x', 'trend1', source=source, color="firebrick", alpha=0.3, line_dash='dashed', legend_label="Tendance Global")
    p.line('x', 'y2', source=source, color="navy", alpha=0.6, legend_label="Europe")
    p.line('x', 'trend2', source=source, color="navy", alpha=0.3, line_dash='dashed', legend_label="Tendance Europe")
    p.line('x', 'y3', source=source, color="green", alpha=0.6, legend_label="France")
    p.line('x', 'trend3', source=source, color="green", alpha=0.3, line_dash='dashed', legend_label="Tendance France")
    hover = HoverTool(
        tooltips=[
            ("Année", "@x"),
            ("Temp_Global", "@y1{0.00}"),
            ("Temp_Europe", "@y2{0.00}"),
            ("Temp_France", "@y3{0.00}"),
            ("Tendance_Global", "@trend1{0.00}"),
            ("Tendance_Europe", "@trend2{0.00}"),
            ("Tendance_France", "@trend3{0.00}")
        ],
        formatters={
            "@x": "numeral",
            "@y1": "numeral",
            "@y2": "numeral",
            "@y3": "numeral",
            "@trend1": "numeral",
            "@trend2": "numeral",
            "@trend3": "numeral"
        },
    )
    p.add_tools(hover)
    p.legend.location = "top_left"
    st.bokeh_chart(p)
    st.write("Grâce à ce graphique (construit à l’aide de la librairie Bokeh), nous pouvons voir que l’augmentation est la même que sur le graphique précédent. Quand bien même nous sommes sur une période de temps plus condensée, on note également une plus franche augmentation à partir de 1980")
    st.write("Pour visualiser cela encore de manière plus interactive, nous avons construit, à l’aide de Plotly, une map intéractive permettant de naviguer dans le temps montrant les anomalies en temps réel. Le dégradé du jaune au rouge nous montre bien que plus le temps avance, plus le rouge est présent sur la map:")
    fig = px.choropleth(data_temp_yearly,
                    locations='Area',
                    locationmode='country names',
                    color='Temperature change',
                    hover_name='Area',
                    animation_frame='Year',
                    color_continuous_scale='RdYlBu_r',
                    projection='natural earth',
                    labels={'Temperature change':'Temperature anomaly (°C)'},
                    title='Global Temperature Anomalies Over Time')

    st.write(fig)

    st.write("Afin de venir renforcer davantage ce que l’on a vu grâce au graphique, il est intéressant de calculer des corrélations entre les différentes zones du monde. Pour cela, nous avons, grâce aux librairie pandas et matplotlib, visualiser ces corrélations sous une carte de chaleur (heatmap):")
    df_numerical = data_temp_average.drop(columns=['24N-90N', '24S-24N', '90S-24S', '64N-90N', '44N-64N', '24N-44N', 'EQU-24N', '24S-EQU', '44S-24S', '64S-44S', '90S-64S'])
    corr = df_numerical.corr()
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", linewidths=.5, ax=ax, cmap='coolwarm')
    st.pyplot(fig)
    st.write("Cette carte de chaleur nous montre bien que le réchauffement est globale et non pas localisée dans une région. En effet, les corrélations fortes entre les hémisphères permettent de conclure qu'il n'y pas réchauffement des températures dans une zone sans que les autres subissent le même sort.")
    st.markdown('<hr style="border: none; border-top: 2px solid #D3D3D3; width: 50%;">', unsafe_allow_html=True)

    st.subheader("II. Concernant les emissions de CO2 :")
    st.write("Maintenant que nous avons constaté qu’il y a bel et bien un réchauffement des températures à travers le temps, il est intéressant de venir intégrer dans la boucle les émissions de CO2 à travers le globe. En effet, il est intéressant de se demander si ces émissions participent à ces augmentations de températures. Mais commençons d’abord par la même démarche que le précédent dataset.")
    st.write("**Est-ce qu’il y a eu une augmentation du CO2 à l’échelle globale à travers le temps ? A partir de quand ?**")
    data_co2 = pd.read_csv("./data/co2_global_non_nettoye.csv", sep=",", encoding='cp1252')
    columns_to_keep = [
    'country',
    'year',
    'iso_code',
    'population',
    'gdp',
    'co2',
    'coal_co2',
    'gas_co2',
    'oil_co2',
    'cement_co2',
    'flaring_co2',
    'other_industry_co2',
    'land_use_change_co2',
    'total_ghg',
    'ghg_per_capita',
    ]
    data_co2 = data_co2[columns_to_keep]
    df_co2_filtered = data_co2[data_co2['year'] >= 1880]
    df_co2_filtered['emissions'] = df_co2_filtered[['cement_co2', 'coal_co2', 'flaring_co2', 'gas_co2', 'oil_co2', 'other_industry_co2']].sum(axis=1)

    df_co2_final_grouped = df_co2_filtered.groupby('year')[['emissions', 'land_use_change_co2']].sum()
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red']
    labels = ['Fossil Fuel and Industry Emissions', 'Land Use Emissions']
    for column, color, label in zip(df_co2_final_grouped.columns, colors, labels):
        plt.plot(df_co2_final_grouped.index, df_co2_final_grouped[column], label=label, color=color)
    plt.xlabel('Année')
    plt.ylabel('Émissions de CO2 (Tonnes)')
    plt.title('Production de CO2 mondiale annuel')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    st.write("Grâce à ce graphique (construit à l’aide de la librairie Matplotlib), nous pouvons voir que la production mondiale de CO2 annuel augmente sévèrement à partir de 1960, synonyme de boom économique.") 
    st.write("On note aussi que cette augmentation de la production de CO2 est principalement dû au secteur industriel. Ce secteur est le plus émetteur étant donné qu’il utilise beaucoup d’énergies fossiles comme le fuel ou le charbon par exemple). A titre de comparaison, les émissions émises naturellement par les sols, stagne dans le temps. Ceci nous montre bien que cette augmentation est directement induite par l’activité humaine sur terre.")
    st.write("Pour visualiser cela encore de manière plus interactive, nous avons construit, à l’aide de Plotly, une map interactive permettant de naviguer dans le temps montrant les émissions de CO2  en temps réel. Le dégradé du jaune au rouge nous montre bien que plus le temps avance, plus ces émissions sont nombreuses, et donc le rouge est présent sur la map. Nous pouvons constater que c’est l’hémisphère Nord et plus particulièrement les USA mais surtout la Chine, plus gros émetteur de Co2 sur la planète:")
    co2_emission_individual_countries = df_co2_filtered.groupby(['country', 'iso_code', 'year'])['co2'].mean().reset_index()
    fig = px.choropleth(co2_emission_individual_countries,
                    locations='iso_code',
                    color='co2',
                    hover_name='country',
                    animation_frame='year',
                    color_continuous_scale='reds',
                    projection='natural earth',
                    labels={'co2':'CO2 emissions (million tonnes)'},
                    title='Global CO2 Emissions Over Time')

    st.write(fig)
    st.write("**Est-ce que ces émissions diffèrent en fonction d’autres paramètres ?**")
    data_co2_par_habitant = pd.read_csv("./data/co2_global_non_nettoye.csv", sep=",", encoding='cp1252')
    df_co2_per_capita_par_pays=data_co2_par_habitant[["country","year","co2_per_capita"]]
    co2_per_capita_par_pays=pd.pivot_table(df_co2_per_capita_par_pays, values='co2_per_capita', index=['year'],columns=['country'])
    co2_per_capita_par_pays=co2_per_capita_par_pays[["North America","South America","Europe","Africa","Asia","Oceania"]]
    co2_per_capita_par_pays=co2_per_capita_par_pays.reset_index()
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    co2_per_capita_par_pays.plot(x="year", y=["North America", "South America", "Europe", "Africa", "Asia", "Oceania"], title="Evolution des émissions de CO2 par habitants", ax=ax)
    st.pyplot(fig)
    st.write("Comme notre dataset est composé d’autres variables représentant des émissions de CO2 en fonction de paramètres, il est intéressant de montrer s’il y a une différence par rapport aux émissions globales.")
    st.write("Il s’avère qu’il y en a une si l’on ramène les émissions de CO2 par habitants ou cette fois-ci, ce n’est plus l’Asie et la Chine qui domine mais, l’Amérique du Nord. Ceci s’explique par le nombre d’habitants moins important dans cette région.")
    st.markdown('<hr style="border: none; border-top: 2px solid #D3D3D3; width: 50%;">', unsafe_allow_html=True)
    st.write("**Maintenant que nous avons constaté que les emissions de CO2 augmentaient au fil du temps tout comme les anomalies de températures, est-ce qu’il y a une corrélation entre ces deux phénomènes ?**")
    st.write("Pour répondre à cette question, nous avons donc mergé nos deux dataset afin de rassembler nos données et nous avons créé ce graphique (à l’aide de Bokeh)")
    window_size = 5
    common_years = set(data_co2['year']).intersection(set(data_temp_average['Year']))
    df_co2_global_common_years = data_co2[data_co2['year'].isin(common_years)]
    df_ZonnAnn_common_years = data_temp_average[data_temp_average['Year'].isin(common_years)].copy()
    df_ZonnAnn_common_years['Glob_smoothed'] = df_ZonnAnn_common_years['Glob'].rolling(window_size, center=True).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_co2_global_common_years['year'], y=df_co2_final_grouped['emissions'], mode='lines', name='Fossil Fuel and Industry Emissions', yaxis='y1'))
    fig.add_trace(go.Scatter(x=df_co2_global_common_years['year'], y=df_co2_final_grouped['land_use_change_co2'], mode='lines', name='Land Use Change Emissions', yaxis='y1'))
    fig.add_trace(go.Scatter(x=df_ZonnAnn_common_years['Year'], y=df_ZonnAnn_common_years['Glob_smoothed'], mode='lines', name='Smoothed Temperature change', yaxis='y2'))
    fig.update_layout(
        title_text="Global CO2 Emissions and Temperature Change Over Time",
        width=800,
        height=600,
        xaxis=dict(domain=[0, 1]),
        yaxis1=dict(title='CO2 Emissions'),
        yaxis2=dict(title='Temperature Change', overlaying='y', side='right')
    )
    st.write(fig)
    st.write("Pour répondre à cette question, nous avons donc mergé nos deux dataset afin de rassembler nos données et nous avons créé ce graphique (à l’aide de Bokeh).")
    st.write("On peut constater clairement qu’il y a un lien entre augmentation des températures et émissions de CO2 sur la planète. La forte hausse commence aussi à la même période. D’ailleurs, on remarque ici que les changements de températures semblent plus corrélés aux émissions provenant du secteur industriel et des énergies fossiles qu’il utilise.")
    
    st.write("Il est donc maintenant intéressant de voir si cette « corrélation » sur le graphique se manifeste dans les chiffres. Nous avons donc calculé, via un calcul de corrélation de Pearson, la corrélation entre les émissions de CO2 et les augmentations de températures dans le temps")
    df_co2_final_grouped=df_co2_final_grouped.reset_index()
    df_temp_emissions = pd.merge(df_ZonnAnn_common_years[['Year', 'Glob']],
                             df_co2_final_grouped[['year', 'emissions']],
                             left_on='Year', right_on='year', how='inner')
    variable_names = {
    "Year": "Année",
    "Glob": "Température Globale",
    "emissions": "Émissions de CO2"
    }
    selected_variables = st.multiselect("Sélectionnez les variables :", list(variable_names.values())) 
    if len(selected_variables) != 2 and len(selected_variables) != 3:
        st.error("Veuillez sélectionner exactement 2 variables.")
    else:
        selected_variables_raw = [key for key, value in variable_names.items() if value in selected_variables]
        if len(selected_variables_raw) == 2:
            variable1, variable2 = selected_variables_raw
            corr, _ = pearsonr(df_temp_emissions[variable1], df_temp_emissions[variable2])
            st.write(f"Corrélation entre {variable_names[variable1]} et {variable_names[variable2]} : **{corr:.2f}**")
            st.write("**La relation est très significative. Par conséquent, nous pouvons conclure que les emissions de CO2 ont une influence forte sur les augmentations de températures dans le temps**")
        elif len(selected_variables_raw) == 3:
            st.error("Veuillez sélectionner exactement 2 variables.")
