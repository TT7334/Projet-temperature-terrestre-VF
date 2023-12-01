import streamlit as st

def conclusion():
    # image
    st.image("./assets/image_conclusion.jpg", use_column_width=True)
    st.markdown(f'<h2 style="text-align: center; text-decoration: underline;">Conclusion</h2>', unsafe_allow_html=True)
    st.markdown('<hr style="border: 2px solid #D3D3D3">', unsafe_allow_html=True)
    
    st.header("Bilan:")
    st.write("A travers ce projet, nous souhaitions principalement savoir si l'analyse des données sur ce sujet nous permettrait d’atteindre des résultats concordants avec la multitude d'études qui existent sur ce sujet. **Et nos résultats partage le même constat, pessimiste.**")
    
    st.write("Grâce à l'analyse statistique des données, nous avons pu répondre aux problématiques posées en début de projet :")
    texte = """
    - Nous avons pu constater le réchauffement climatique : **c'est une réalité indiscutable**. Cela augmente depuis la révolution industrielle de 1880, et s'accélère à partir de 1975, de manière plus forte encore dans l'hémisphère nord, là où l'industrialisation est la plus intensive.
    - Les émissions de CO2 ont **une forte influence** sur la hausse des températures. Elles sont étroitement liées.
    - **Nos prédictions ne sont pas optimistes**. D'après notre modèle Prophet, la température moyenne globale augmentera de près de 2°C dans les 50 prochaines années, causées par une augmentation constante des émissions de CO2 et de toutes les autres émissions de gaz à effet de serre qui ont un lien de corrélation avec ces changements de températures. Ces résultats rejoignent ceux de la majorité des études disponibles sur le changement climatique.
    """
    st.markdown(texte)


    st.header("Sujets d'ouverture:")
    st.write("En guise d'ouverture, il serait intéressant de regarder **comment évoluent les différences de températures** en prévoyant des **diminutions (ou augmentations) des émissions de gaz à effet de serre.**")
    st.write("Le comportement des **températures sera probablement différent** et permettra de donner à l'humanité des **objectifs d'emissions** à atteindre pour vivre de manière pérenne sur notre planète.")
