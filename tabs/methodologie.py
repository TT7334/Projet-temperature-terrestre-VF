import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import gaussian_kde
from scipy.stats import pearsonr
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet

def methodo():
    st.image("./assets/methodologie.jpg", use_column_width=True)
    # Sous-titre "methodo"
    st.markdown(f'<h2 style="text-align: center; text-decoration: underline;">Méthodologie de la modélisation</h2>', unsafe_allow_html=True)
    st.write("Maintenant que nous avons constaté qu'il y avait bien un réchauffement des températures à l'échelle de la planète et que cette augmentation est fortement influencée par les emmissions de CO2 (principalement produites par l'industrialisation), essayons de prédire les changements de températures et celles du CO2 dans notre modèle")
    st.markdown('<hr style="border: 2px solid #D3D3D3">', unsafe_allow_html=True)
    # La première étape est de faire des prédictions sur le CO2
    st.write("Voici donc la méthodologie que nous avons adoptée pour mener à bien notre projet:")
    texte = """
    - Nous nous sommes donc d'abord, nous nous sommes penchés sur les changements de températures:
        - Nettoyage complet de la base de données avec nottament la gestion des valeurs manquantes et l'encodage des variables
        - Test de différents modèles pour ne garder que le plus performant.
    - Une fois chose faite, nous nous sommes penchés sur les émissions de CO2:
        - Nettoyage complet de la base de données avec nottament la gestion des valeurs manquantes et l'encodage des variables
        - Test de différents modèles pour ne garder que le plus performant.
    """
    st.markdown(texte)
    st.markdown('<hr style="border: none; border-top: 2px solid #D3D3D3; width: 50%;">', unsafe_allow_html=True)
    st.subheader("I. Température :")
    st.write("**Nettoyage de la base de données:**")
    st.write("Dans la partie précédente preprocessing, nous avons déjà appliqué la fonction de suppression des valeurs manquantes et filtré à partir de l'année 1961. Voici le dataset en sortie:")
    df_temp= pd.read_csv('./data/temperature_nettoyé_avant_gestion_NAN.csv', sep=",")
    st.write(df_temp)
    st.write("Il est important de préciser ici quelques points:")
    texte = """
    - Contrairement à la partie Datavisualisation, nous avons gardé les codes de chaque pays pour nous éviter de les encoder par la suite. Donc chaque pays à un code attribué.
    - Nous n'avons gardé que la modalité "meteorogical year" qui permet d'avoir une valeur par année pour chaque pays.
    """
    st.markdown(texte)
    st.write("Nous avons donc un dataset plus clair avec autant de code qu'il n'y a de pays dans le dataset, les années de mesures allant de 1961 à 2020 pour chaque pays et leur valeurs de température associées.")
    st.write("Il ne reste plus qu'à traiter les valeurs manquantes restantes après cette étape de nettoyage.")
    missing_values = df_temp.isna().sum()
    st.write(missing_values)
    st.write("Nous voyons ici qu'il reste des valeurs manquantes à hauteur de **481** dans la variable Value, soit 0.2 pour cent du dataset")
    st.write("Nous allons donc appliquer la méthode des KNN plus proches voisins afin de remplir ces valeurs:")
    knn_imputer = KNNImputer(n_neighbors=3)
    df_temp.iloc[:, 2:] = knn_imputer.fit_transform(df_temp.iloc[:, 2:])
    missing_values = df_temp.isna().sum()
    st.write(missing_values)
    st.write("Nous avons donc maitenant un Dataset nettoyé et prêt à être modélisé:")
    st.write(df_temp)
    
    st.write("**Différents modèles utilisés:**")
    st.write("Afin de faire les prédictions les plus précises possibles, nous avons décidé de tester plusieurs modèles de ML, comparer leur performances pour ne retenir que le plus performant.")
    st.write("voici les 3 modèles testés:")
    texte = """
    - Une régression linéaire
    - Une série temporelle avec le modèle SARIMAX
    - Une série temporelle avec le modèle Prophet
    """
    st.markdown(texte)
    st.markdown("<u>*A. La régréssion linéaire:*</u>", unsafe_allow_html=True)
    #je réinitialise l'index pour avoir quelque chose qui commence à 0
    df_temp=df_temp.reset_index()
    df_temp = df_temp.drop(columns=["index"])
    #Je sépare mon jeu sous format 80/20 avec 80% pour l'entrainement et 20% pour le test.La découpe se fait non pas alétoirement mais par année :
    texte = """
    - Nous isolons notre variable cible "Value" du reste du dataset
    - Nous séparons notre jeu de données non pas de manière aléatoire mais en fonction des années pour respecter l'aspect chronologique de notre dataset. Les années de 1961 à 2008 (soit 80%) constituera notre jeu d'entrainement. Notre jeu de test (20%) ira de l'année 2009 à 2020
    - Nous entrainons ensuite notre modèle sur les dataframe X_train et y_train
    """
    st.markdown(texte)
    train = df_temp[df_temp['Year'] <= 2008]
    test = df_temp[(df_temp['Year'] >= 2009) & (df_temp['Year'] <= 2020)]
    #Maintenant, j'isole mes variables explicatives et ma variable cible qui est value.
    X_train = train[['Year',"Area Code (FAO)"]]
    y_train = train['Value']
    X_test = test[['Year',"Area Code (FAO)"]]
    y_test = test['Value']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader("X train")
        st.write(X_train)
    with col2:
        st.subheader("Y train")
        st.write(y_train)
    with col3:
        st.subheader("X test")
        st.write(X_test)
    with col4:
        st.subheader("Y test")
        st.write(y_test)
        
    #Entrainement du modèle
    model = LinearRegression()
    model.fit(X_train, y_train)
    texte = """
    - A la suite de cela, nous souhaitons visualiser si nos prédictions pour y sont cohérentes avec les valeurs de y_test:
    """
    st.markdown(texte)
    # Prédiction avec l'ensemble de test.
    y_pred_test = model.predict(X_test)
    # Visualisation des résultats de prédiction afin de voir si elles sont cohérentes avec les véritables données.
    X_train_year = X_train["Year"]
    X_test_year = X_test["Year"]
    plt.figure(figsize=(10, 5))
    plt.scatter(X_train_year, y_train, color='blue', label='Training Data')
    plt.scatter(X_test_year, y_test, color='green', label='Test Data')
    plt.scatter(X_test_year, y_pred_test, color='red', label='Predictions')
    plt.legend()
    plt.title('Temperature Prediction')
    plt.xlabel('Year')
    plt.ylabel('Temperature Anomaly')
    st.pyplot(plt)
    st.write("Nous voyons ici que la trajectoire, la tendance des prédictions suit celle de y_test en verte. Nous vérifierons la véracité de ce propos lors du calcul de la performance du modèle")
    texte = """
    - Maintenant, évaluons notre modèle pour pouvoir le comparer avec les autres:
    """
    st.markdown(texte)
    # Évaluation du modèle avec l'ensemble de test.
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    #Affichage des résultats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px;'>"
                    "<p><strong>MSE (Mean Squared Error)</strong></p>"
                    f"<p>{mse:.2f}</p>"
                    "</div>", unsafe_allow_html=True)
    with col1:
        st.markdown("<div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px;'>"
                    "<p><strong>RMSE (Root Mean Squared Error)</strong></p>"
                    f"<p>{rmse:.2f}</p>"
                    "</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px;'>"
                    "<p><strong>MAE (Mean Absolute Error)</strong></p>"
                    f"<p>{mae:.2f}</p>"
                    "</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px;'>"
                    "<p><strong>R^2 (Coefficient of Determination)</strong></p>"
                    f"<p>{r2:.2f}</p>"
                    "</div>", unsafe_allow_html=True)
    
    
    
    st.write("Visiblement, les résultats de cette régréssion ne sont pas très bons avec une RMSE à 0.58 mais surtout un coéfficient de determination **négatif**. Cela montre que le modèle prend mal en compte la variance de la série.")
    st.write("Essayons donc un autre modèle et nottament des modèles ARIMA et SARIMAX, censé capter de meilleure manière la variance de nos données:")
    
    st.markdown("<u>*B. Le modèle SARIMAX:*</u>", unsafe_allow_html=True)
    texte = """
    - Tout d'abord, il faut convertir notre année en indice temporel 
    - Ensuite, pour mettre en place notre modèle, il ne faudra choisir qu'un seul pays. Nous décidons de choisir la France pour ensuite appliquer le modèle à notre dataset.
    """
    st.markdown(texte)
    # D'abord, il faut convertir l'année en indice temporel
    df_temp["Year"] = pd.to_datetime(df_temp["Year"], format='%Y')
    df_temp.set_index("Year", inplace=True)
    df_france = df_temp[df_temp['Area Code (FAO)'] == 68]
    #Visualisation de la série temporelle
    st.write("Visualisons d'abord la série temporelle:")
    plt.figure(figsize=(12,6))
    plt.plot(df_france.index, df_france['Value']) 
    plt.title('Température en France de 1961 à 2020')
    plt.xlabel('Année')
    plt.ylabel('Différence de température')
    plt.grid(True)
    st.pyplot(plt)
    
    st.write("Maintenant, il faut pouvoir décomposer la série afin d'analyser la tendance, la saisonnalité et les résidus:")
    #Décomposition de la série pour analyser la tendance, la saisonnalité et les résidus
    decomposition = seasonal_decompose(df_france["Value"],period=1)
    # Affichage des composants décomposés
    fig = decomposition.plot()
    st.pyplot(plt)
    st.write("À partir de cette décomposition, nous pouvons conclure que la série temporelle peut être modélisée comme un modèle additif, Les composants saisonniers semblent s'ajouter (et non se multiplier) à la tendance générale. Cela est confirmé par le fait que l'amplitude de la saisonnalité ne semble pas changer de manière significative au fil du temps.")
    st.write("La prochaine étape de notre processus est de voir si la série est stationnaire et si ce n'est pas le cas, il faudra la rendre stationnaire:")
    # Tester la stationnarité avec le test adfuller
    result = adfuller(df_france['Value'])
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', result[1])
    st.write("la série temporelle de la différence de température pour la France est probablement **non-stationnaire**. En effet, la valeur ADF est positive et la p-value supérieur à 5%. Pour la modélisation, il serait judicieux d'essayer de différencier pour la rendre stationnaire avant de poursuivre.")
    #Donc pour rendre stationnaire notre série temporelle, nous allons la différencier 
    df_france_diff = df_france['Value'].diff().dropna()
    # Vérifier visuellement si la série différenciée semble stationnaire
    plt.figure(figsize=(12,6))
    plt.plot(df_france_diff)
    plt.title('Différence de température différenciée pour la France')
    plt.xlabel('Année')
    plt.ylabel('Différence de température différenciée')
    plt.grid(True)
    st.pyplot(plt)
    #Pour voir si la série est bien devenue stationnaire, on peut refaire un test de adfuller avec cette série différenciée
    result = adfuller(df_france_diff)
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', result[1])
    texte = """
    - Une statistique ADF aussi négative est un indicateur fort de la stationnarité. De plus, la p-value est bien inférieure au seuil commun de 0,05.
    - En conclusion, ces résultats indiquent que la série différenciée est maintenant stationnaire.
    """
    st.markdown(texte)
    st.write("#Pour pouvoir utiliser le modèle, il faut pouvoir lui donner les bons paramètres de p et q. Pour cela, nous allons utiliser les fonctions d'auto-corrélation (ACF) et d'auto-corrélation partielle (PACF).")
    #Visualiser ACF et PACF
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plot_acf(df_france_diff, ax=plt.gca(), lags=30)
    plt.subplot(122)
    plot_pacf(df_france_diff, ax=plt.gca(), lags=28)
    plt.tight_layout()
    st.pyplot(plt)
    texte = """
    - ACF (Autocorrelation Function) : Le graphique ACF montre que l'autocorrélation pour le premier lag est significative, et après cela, elle reste en dessous de la bande de confiance. Cela suggère que q = 1
    - PACF (Partial Autocorrelation Function) : Le graphique PACF indique que la valeur partielle d'autocorrélation est significative pour le premier lag, puis elle tombe en dessous de la bande de confiance. Cela suggère que p = 1.
    - Ainsi, en se basant sur ces graphiques, le modèle ARIMA serait :
        - p = 1 (pour la composante AR)
        - d = 1 (j'ai déjà différencié votre série une fois)
        - q = 1 (pour la composante MA)
    """
    st.markdown(texte)
    from statsmodels.tsa.arima.model import ARIMA
    # Création du modèle ARIMA
    model = ARIMA(df_france['Value'], order=(1,1,1))
    result = model.fit()
    # Ajuster le modèle ARIMA(0,1,1)
    model = ARIMA(df_france['Value'], order=(0,1,1))
    result2 = model.fit()
    # Création d'un accordéon pour chaque modèle ARIMA
    def summary_to_dataframe(arima_results):
        coef_table_html = arima_results.summary().tables[1].as_html()
        return pd.read_html(coef_table_html, header=0, index_col=0)[0]
    df_result1 = summary_to_dataframe(result)
    df_result2 = summary_to_dataframe(result2)
    with st.expander("Modèle ARIMA(1,1,1)"):
        st.dataframe(df_result1)
    with st.expander("Modèle ARIMA(0,1,1)"):
        st.dataframe(df_result2)
    texte = """
    - Après analyse des résultats du paramètre, le modèle ARIMA(1,1,1) semble être un ajustement approprié pour les données. Cependant,  (AR) n'est pas statistiquement significatif.
    - Nous ajustons alors les paramètres du modèle ARIMA(0,1,1)
    - En comparant les résultats, le modèle **ARIMA(0,1,1) semble être un meilleur choix pour les données**, car il a des critères d'information légèrement plus bas tout en conservant de bonnes propriétés résiduelles. Cependant, la différence entre les deux modèles n'est pas très grande.
    """
    st.markdown(texte)
    st.write("Maintenant que nous avons trouvé les paramètres de notre modèle ARIMA, passons au modèle SARIMAX en ajoutant les ordres saisonniers")
    st.write("Nous avons testé plusieurs ordres et il s'avère que le plus performant est l'ordre 25, signifiant que, approximativement, les valeurs se 'répètent' tous les 25 ans")
    texte = """
    - Le modèle SARIMAX le plus cohérent étant trouvé, il est temps maintenant de voir si celui-ci est performant. 
    - Nous séparons le jeu de données en fonction des années, comme nous l'avons fait précédemment dans notre régression
    - Nous entrainons le modèle.
    """
    st.markdown(texte)
    train_fr = df_france.loc['1961-01-01':'2008-12-31', 'Value']
    test_fr = df_france.loc['2009-01-01':'2020-12-31', 'Value']
    model = SARIMAX(train_fr, 
            order=(0, 1, 1),             
            seasonal_order=(0, 1, 1, 25)) 
    # Entraînement du modèle
    results = model.fit()
    # Prévisions
    forecast_fr = results.get_forecast(steps=len(test_fr))
    forecast_fr_mean = forecast_fr.predicted_mean
    # Création de l'intervalle de confiance pour les prévisions
    confidence_intervals = forecast_fr.conf_int()
    #Visualisation de la performance du modèle avec y_test
    st.write("En comparant nos données prédites par rapport à y_test, nous constatons que nos prédictions se rapprochent des vraies données. **La variance a donc été mieux prise en compte par rapport à notre régression linéaire**")
    plt.figure(figsize=(12,6))
    plt.plot(train_fr.index, train_fr, label="Training", color='blue') 
    plt.plot(test_fr.index, test_fr, label="Actual", color='black')     
    plt.plot(test_fr.index, forecast_fr_mean, label="Forecast", color='orange', linestyle='--') 
    # Tracé de l'intervalle de confiance
    plt.fill_between(test_fr.index, 
                    confidence_intervals.iloc[:, 0], 
                    confidence_intervals.iloc[:, 1], color='pink', alpha=0.3)
    plt.title("Temperature Forecast for France with SARIMA")
    plt.xlabel("Year")
    plt.ylabel("Temperature Anomaly")
    plt.legend()
    st.pyplot(plt)
   
    st.write("Vérifions cela en vérifiant la performance du modèle. Celui-ci devrait être meilleur que notre régréssion linéaire: ")
    #Affichage des résultats
    mse = mean_squared_error(test_fr, forecast_fr_mean)
    rmse = mean_squared_error(test_fr, forecast_fr_mean, squared=False)
    mae = mean_absolute_error(test_fr, forecast_fr_mean)
    r2 = r2_score(test_fr, forecast_fr_mean)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px;'>"
                    "<p><strong>MSE (Mean Squared Error)</strong></p>"
                    f"<p>{mse:.2f}</p>"
                    "</div>", unsafe_allow_html=True)
    with col1:
        st.markdown("<div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px;'>"
                    "<p><strong>RMSE (Root Mean Squared Error)</strong></p>"
                    f"<p>{rmse:.2f}</p>"
                    "</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px;'>"
                    "<p><strong>MAE (Mean Absolute Error)</strong></p>"
                    f"<p>{mae:.2f}</p>"
                    "</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px;'>"
                    "<p><strong>R^2 (Coefficient of Determination)</strong></p>"
                    f"<p>{r2:.2f}</p>"
                    "</div>", unsafe_allow_html=True)
    st.write("Nous le constatons, **les résulats sont meilleurs** surtout au niveau du coefficient de détermination qui est repassé dans le **positif avec une valeur de 0.35**")
    
    st.write("Essayons maintenant avec un dernier modèle, l'algorithme Prophet, développé par Facebook et censé être le plus performant pour traiter les séries temporelles")

    st.markdown("<u>*C. Le modèle Prophet*</u>", unsafe_allow_html=True)
    st.write("Nous avons ensuite appliqué la même méthodologie que notre modèle SARIMAX mais en utilisant le modèle prophet de facebook afin de voir si nous avions des performances plus satisfaisantes, et voici les résultats:")
    st.image("./assets/composant_prophet.png", use_column_width=True)
    
    #Affichage des résultats
    mse_prophet = 0.02552076874679013
    rmse_prophet = 0.1597522104597934
    mae_prophet = 0.13353782950574777
    r2_prophet = 0.7960427105311697

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px;'>"
                    "<p><strong>MSE (Mean Squared Error)</strong></p>"
                    f"<p>{mse_prophet:.2f}</p>"
                    "</div>", unsafe_allow_html=True)
    with col1:
        st.markdown("<div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px;'>"
                    "<p><strong>RMSE (Root Mean Squared Error)</strong></p>"
                    f"<p>{rmse_prophet:.2f}</p>"
                    "</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px;'>"
                    "<p><strong>MAE (Mean Absolute Error)</strong></p>"
                    f"<p>{mae_prophet:.2f}</p>"
                    "</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px;'>"
                    "<p><strong>R^2 (Coefficient of Determination)</strong></p>"
                    f"<p>{r2_prophet:.2f}</p>"
                    "</div>", unsafe_allow_html=True)
    
    st.write("Avec prophet, **les résulats sont encore meilleurs** que notre modèle Sarimax sur tous les indicateurs")

    st.write("**Pour conclure, parmis les différents modèles que nous avons testé, le modèle Prophet est le plus performant pour prédire nos températures. C'est celui que nous allons donc garder pour la partie suivante**")


    st.subheader("II. CO2 :")
    st.write("**Nettoyage de la base de données:**")
    st.write("Dans la partie précédente preprocessing, nous avons déjà appliqué la fonction de suppression des valeurs manquantes, filtré à partir de l'année 1961 et supprimer les variables qui ne nous intéressaient pas. Voici le dataset en sortie:")
    df_co2= pd.read_csv('./data/co2_nettoyé_avant_gestion_NAN.csv', sep=",")
    st.write(df_co2)
    st.write("Il est important de préciser ici quelques points:")
    texte = """
    - Contrairement au dataset sur les anomalies de température, nous n'avons pas les codes des pays. Il faudra donc les encoder. 
    - En plus, nous avons une variable de population dans ce dataset. Il faudra donc standardiser ces données pour les ramener à la même échelle que les valeurs de CO2.
    """
    st.markdown(texte)
    st.write("Commençons, avant cela, par l'imputation des valeurs manquantes:")
    missing_values2 = df_co2.isna().sum()
    st.write(missing_values2)
    st.write("Nous voyons ici qu'il reste des valeurs manquantes sur les variable 'population' et 'Co2' à hauteur, respectivement, de **1 908** et **1 851**")
    st.write("Nous allons donc appliquer la méthode des KNN plus proches voisins afin de remplir ces valeurs:")
    columns_to_impute = ['population', 'co2']
    data_to_impute = df_co2[columns_to_impute]
    imputer2 = KNNImputer(n_neighbors=3)
    imputed_data = imputer2.fit_transform(data_to_impute)
    df_co2[columns_to_impute] = imputed_data
    missing_values3 = df_co2.isna().sum()
    st.write(missing_values3)
    st.write("Nous avons donc maitenant un Dataset nettoyé et prêt à être modélisé:")
    st.write(df_co2)

    st.write("**Différents modèles utilisés:**")
    st.write("Afin de faire les prédictions les plus précises possibles(de la même manière que le dataset des températures), nous avons décidé de tester plusieurs modèles de ML, comparer leur performances pour ne retenir que le plus performant.")
    st.write("voici les 3 modèles testés:")
    texte = """
    - Une régréssion linéaire
    - Un Random Forest
    - Une série temporelle avec le modèle Prophet
    """
    st.markdown(texte)
    st.markdown("<u>*A. La régréssion linéaire:*</u>", unsafe_allow_html=True)
    st.write("Avant de commencer, nous réinitialisons l'index pour retrouver de la chrnologie. De plus, commé évoqué, nous devons encoder la variable 'country' puisque c'est un float. Nous utilisons pour ça la méthode du 'One Hot Encoder':")
    df_co2= pd.read_csv('./data/co2_nettoyé.csv', sep=",")
    st.write(df_co2)
    #Je sépare mon jeu sous format 80/20 avec 80% pour l'entrainement et 20% pour le test.La découpe se fait non pas alétoirement mais par année :
    texte = """
    - Nous isolons notre variable cible "co2" du reste du dataset
    - Nous séparons notre jeu de données non pas de manière aléatoire mais en fonction des années pour respecter l'aspect chronologique de notre dataset. Les années de 1961 à 2008 (soit 80%) constituera notre jeu d'entrainement. Notre jeu de test (20%) ira de l'année 2009 à 2021
    - Une fois le jeu séparé, il est mthodologiquement bon de standardiser certaines variables pour les mettre à l'échelle par rapport au reste. Nous décidons de standardiser la variable "population".
    - Nous pouvons maintenant entrainer notre modèle sur les dataframe X_train et y_train
    """
    st.markdown(texte)
    feats = df_co2.drop('co2', axis=1)
    target = df_co2['co2']
    cutoff_year = 2008
    X_train = feats[df_co2['year'] <= cutoff_year]
    X_test = feats[df_co2['year'] > cutoff_year]
    y_train = target[df_co2['year'] <= cutoff_year]
    y_test = target[df_co2['year'] > cutoff_year]
    scaler = StandardScaler()
    scaler.fit(X_train[['population']])
    # Transformer les valeurs 'population' dans les ensembles d'entraînement et de test
    X_train['population'] = scaler.transform(X_train[['population']])
    X_test['population'] = scaler.transform(X_test[['population']])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader("X train")
        st.write(X_train)
    with col2:
        st.subheader("Y train")
        st.write(y_train)
    with col3:
        st.subheader("X test")
        st.write(X_test)
    with col4:
        st.subheader("Y test")
        st.write(y_test)
    #j'applique un modèle de régression linéaire:
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    #Prediction sur les données test
    y_pred = regressor.predict(X_test)
    texte = """
    - A la suite de cela, nous souhaitons visualiser si nos prédictions pour y sont cohérentes avec les valeurs réelles de y_test:
    """
    st.markdown(texte)
    fig = plt.figure(figsize = (8,8))
    pred_test = regressor.predict(X_test)
    plt.scatter(pred_test, y_test, c='green')
    plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color = 'red')
    plt.xlabel("prediction")
    plt.ylabel("vrai valeur")
    plt.title('Régression Linéaire pour la prédiction des émissions de co2')
    st.pyplot(plt)
    st.write("Nous voyons ici que lorsque les valeurs d'émissions sont faibles, le modèle est plutôt efficace en suivant la ligne de régréssion. En revanche, plus les émissions sont importantes, plus le modèle s'affaiblit")
    texte = """
    - Maintenant, évaluons notre modèle pour pouvoir le comparer avec les autres:
    """
    st.markdown(texte)
    # Évaluation du modèle avec l'ensemble de test.
    mse = mean_squared_error(y_test, pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, pred_test)
    r2 = r2_score(y_test, pred_test)
    #Affichage des résultats
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px;'>"
                    "<p><strong>MSE (Mean Squared Error)</strong></p>"
                    f"<p>{mse:.2f}</p>"
                    "</div>", unsafe_allow_html=True)
    with col1:
        st.markdown("<div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px;'>"
                    "<p><strong>RMSE (Root Mean Squared Error)</strong></p>"
                    f"<p>{rmse:.2f}</p>"
                    "</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px;'>"
                    "<p><strong>MAE (Mean Absolute Error)</strong></p>"
                    f"<p>{mae:.2f}</p>"
                    "</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px;'>"
                    "<p><strong>R^2 (Coefficient of Determination)</strong></p>"
                    f"<p>{r2:.2f}</p>"
                    "</div>", unsafe_allow_html=True)
    st.write("Visiblement, les résultats de cette régréssion sont plutôt corrects avec notamment un coefficient de détermination proche de 1.")
    st.write("Essayons tout de même d'améliorer cela avec d'autres modèles plus précis comme l'arbre de décision ou le RandomForest")
    
    st.markdown("<u>*B. Le Random Forest et le l'abre de décision:*</u>", unsafe_allow_html=True)
    st.write("Comme la démarche est identique à la régréssion, nous affichons directement ici les deux courbes de régression ainsi qu'un tableur comparant les deux modèles pour voir quel est le plus performant:")
    #Mise en place du modèle du RandomForestRegressor
    from sklearn.ensemble import RandomForestRegressor
    regressor_random_forest = RandomForestRegressor(random_state=42) 
    regressor_random_forest.fit(X_train,y_train)
    #Mise en place du modèle d'arbre de décision
    from sklearn.tree import DecisionTreeRegressor
    regressor_decision_tree = DecisionTreeRegressor(random_state=42) 
    regressor_decision_tree.fit(X_train, y_train)
    #Prédictions sur les deux modèles
    pred_test_rf = regressor_random_forest.predict(X_test)
    pred_test_ad = regressor_decision_tree.predict(X_test)
    #Visualisation des régressions 
    col1, col2 = st.columns(2)

    # Premier graphique : Random Forest
    with col1:
        fig, ax = plt.subplots(figsize = (8,8))
        ax.scatter(pred_test_rf, y_test, c='green')
        ax.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color = 'red')
        ax.set_xlabel("Prédictions")
        ax.set_ylabel("Vraies valeurs")
        ax.set_title('Random Forest pour la prédiction des émissions de CO2')
        st.pyplot(fig)

    # Deuxième graphique : Arbre de décision
    with col2:
        fig, ax = plt.subplots(figsize = (8,8))
        ax.scatter(pred_test_ad, y_test, c='green')
        ax.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color = 'red')
        ax.set_xlabel("Prédictions")
        ax.set_ylabel("Vraies valeurs")
        ax.set_title('Arbre de décision pour la prédiction des émissions de CO2')
        st.pyplot(fig)
    st.write("Les deux modèles semblent globalement identique. Ce que l'on peut dire, c'est que les points sont plus proches de la ligne de régression et surtout pour les plus grandes émissions. Cela signifie que les modèles sont plus performants que la régréssion linéaire")
    
    st.write("Regardons maintenant cela par les chiffre en analysant les performances des modèles respectifs:")
    #Evaluation des performances du modèle
    # MSE randomforest
    mse_rf = mean_squared_error(y_test, pred_test_rf)
    # RMSE randomforest
    rmse_rf = mean_squared_error(y_test, pred_test_rf, squared=False)
    # MAE randomforest
    mae_rf = mean_absolute_error(y_test, pred_test_rf)
    # R-squared
    r2_rf = r2_score(y_test, pred_test_rf)

    # MSE arbredecision
    mse_ad = mean_squared_error(y_test, pred_test_ad)
    # RMSE abredecision
    rmse_ad = mean_squared_error(y_test, pred_test_ad, squared=False)
    # MAE abredecision
    mae_ad = mean_absolute_error(y_test, pred_test_ad)
    # R-squared arbredecision
    r2_ad = r2_score(y_test, pred_test_ad)
    # Creation d'un dataframe pour comparer les metriques des deux algorithmes 
    data = {'MSE': [mse_rf, mse_ad],
            'RMSE': [rmse_rf, rmse_ad],
            'MAE': [mae_rf,mae_ad],
            'R2': [r2_rf,r2_ad]}
    # Creer DataFrame
    df_joined = pd.DataFrame(data, index = ['Random Forest','Decision Tree'])
    st.write(df_joined)
    st.write("Nous constatons que les RMSE sont plus faibles, ce qui est un bon signe. TOut comme les coefficients de détermination qui se rapprochent davantage de 1. Bon signe également. ")
    st.write("Les deux derniers modèles sont donc plus précis et de meilleurs qualité que la régréssion. En l'état, le modèle le plus performant sur nos données est le random forest.")
    st.write("Essayons tout de même d'appliquer le modèle prophet qui était le modèle le plus performant pour les température")

    st.markdown("<u>*C. Le modèle Prophet*</u>", unsafe_allow_html=True)
    st.write("Pour ce modèle, au même titre que le modèle SARIMAX, nous ne pouvons appliquer le modèle à l'ensemble du dataset, il faut pouvoir entrainer le modèle sur un pays/localisation. Par conséquent, nous choisissons la modalité 'World', pour rester cohérent." )
    world_data= pd.read_csv('./data/co2_nettoyé_world_prophet.csv', sep=",")
    st.write(world_data)
    st.write("A noter ici que nous ne sommes pas dans l'obligation d'encoder puisque nous sommes directement sur un dataframe filtré à la modalité 'World' de la variable country. Et 'ds' étant les années, 'y' les émissions de CO2." )
    st.write("Nous divisons maintenant notre dataframe en un ensemble de test un ensemble d'entrainement, toujours en séparant en fonction des années avec une répartition de 80%/20%:")
    train_size_world = int(0.80 * len(world_data))
    train_world = world_data.iloc[:train_size_world]
    test_world = world_data.iloc[train_size_world:]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Train")
        st.write(train_world)
    with col2:
        st.subheader("Test")
        st.write(test_world)
    st.write("Prophet est optimisé pour la modélisation de séries temporelles univariables. Cela signifie qu'il prend en entrée une série temporelle consistant en deux colonnes : une pour le temps et une pour la variable cible. Nous allons donc mettre de côté la variable 'population' pour le moment.")
    texte = """
    - Nous entrainons donc notre modèle sur les données d'entrainement
    - Nous regardons maintenant la performance de notre modèle en reprenant les différents indicateurs que nous utilisons depuis le début
    """
    st.markdown(texte)
    prophet_world = Prophet(yearly_seasonality=True)
    prophet_world.fit(train_world[['ds', 'y']])
    train_predictions_world = prophet_world.predict(train_world[['ds']])
    rmse_train_world = np.sqrt(mean_squared_error(train_world['y'], train_predictions_world['yhat']))
    mse_train_world = mean_squared_error(train_world['y'], train_predictions_world['yhat'])
    mae_train_world = mean_absolute_error(train_world['y'], train_predictions_world['yhat'])
    r2_train_world = r2_score(train_world['y'], train_predictions_world['yhat'])
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px;'>"
                    "<p><strong>MSE (Mean Squared Error)</strong></p>"
                    f"<p>{mse_train_world:.2f}</p>"
                    "</div>", unsafe_allow_html=True)
    with col1:
        st.markdown("<div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px;'>"
                    "<p><strong>RMSE (Root Mean Squared Error)</strong></p>"
                    f"<p>{rmse_train_world:.2f}</p>"
                    "</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px;'>"
                    "<p><strong>MAE (Mean Absolute Error)</strong></p>"
                    f"<p>{mae_train_world:.2f}</p>"
                    "</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='background-color: #f4f4f4; padding: 10px; border-radius: 10px;'>"
                    "<p><strong>R^2 (Coefficient of Determination)</strong></p>"
                    f"<p>{r2_train_world:.2f}</p>"
                    "</div>", unsafe_allow_html=True)
    st.write("Cette fois encore, nous nous rendons compte que le modèle le plus performant pour prédire l'évolutions des émissions de CO2 est le **Prophet**")
    st.write("**En effet, la MSE, la RMSE et le coefficient de détermination sont meilleurs.**")

    st.write("Après ces nombreux tests, nous concluons donc que le modèle le plus performant est le Prophet, que ce soit pour les températures ou le CO2.")
    st.write("**Par conséquent, nous allons utiliser ce modèle pour faire nos prédictions dans la dernière partie**")

