import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm
import folium
from folium.plugins import HeatMap
import os
import base64
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import calendar
import gdown

# === CONFIGURATION PAGE ===
st.set_page_config(
    page_title="Biodiversité Grand Est 2025",
    page_icon="leaf",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CSS PERSONNALISÉ ===
st.markdown("""
<style>
    .main { background: linear-gradient(to bottom, #f0f8f0, #e6f2e6); }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    h1, h2, h3 { color: #1e3d1e; font-weight: 600; }
    .css-1d391kg { padding-top: 1rem; }
    .stSelectbox, .stMultiselect {background-color: #ffffff; border-radius: 8px; }
    .stPlotlyChart { border: 1px solid #d0e0d0; border-radius: 10px; }
    .highlight {padding: 15px; border-radius: 10px; border-left: 5px solid #28a745; }
</style>
""", unsafe_allow_html=True)

#background-color: #d4edda;

# === CHARGEMENT DES DONNÉES ===
@st.cache_data
def load_data():
    # df = pd.read_csv(r"D:\Dataviz\biodiv_grand_est_merger.csv", parse_dates=['dateObservation'])
    # df = pd.read_csv("https://drive.google.com/uc?export=download&id=1m_KQI34v87PzPx30xMIXpbFs36Wcmrnl", parse_dates=["dateObservation"])
    
    # communes_grand_est = gpd.read_file(r"02_Donnees_Secondaires/communes-grand-est.geojson")
    # departements_grand_est = gpd.read_file(r"02_Donnees_Secondaires/departements-grand-est.geojson")

    def download_geojson(drive_id, output_name):
        if not os.path.exists(output_name):
            url = f"https://drive.google.com/uc?id={drive_id}"
            gdown.download(url, output_name, quiet=False)
        return output_name
    
    # Télécharger et charger
    communes_file = download_geojson("1wo29QyCD-KqnSIw6c9z7WJq_J0iYJO0M", "communes-grand-est.geojson")
    departements_file = download_geojson("1mlOePPCpFTtmevvXnXhbKhnIS4kVk7CL", "departements-grand-est.geojson")
    
    communes_grand_est = gpd.read_file(communes_file)
    departements_grand_est = gpd.read_file(departements_file)
    
    communes = pd.read_csv(
        r"02_Donnees_Secondaires/communes.csv",
        sep=';',  # ou ',' selon ton fichier
        engine='python'  # moteur plus tolérant
    )
    
    file_id = "1m_KQI34v87PzPx30xMIXpbFs36Wcmrnl"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "biodiv_grand_est_merger.csv"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    df = pd.read_csv(output, parse_dates=["dateObservation"])

        
    # Nettoyage
    for col in ["population 2025", 'urbain', 'agricole', 'naturel', 'humide', 'eau']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['mois'] = df['dateObservation'].dt.month
    df['annee'] = df['dateObservation'].dt.year
    df['saison'] = df['mois'].apply(lambda x: 
        'Printemps' if x in [3,4,5] else
        'Été' if x in [6,7,8] else
        'Automne' if x in [9,10,11] else 'Hiver'
    )
    return df, communes_grand_est, departements_grand_est, communes

df, communes_grand_est, departements_grand_est, communes = load_data()

# === SIDEBAR : FILTRES GLOBAUX ===
st.sidebar.image("vignette.jpg", width=200)

# pages = ["Accueil & Storytelling", "Analyses Temporelles", "Analyses Spatiales", "Corrélations & Habitat", "Biais & Normalisations", "Interprétations", "Synthèse & Recommandations"]
pages = ["Accueil & Storytelling", "Analyses Temporelles", "Analyses Spatiales", "Corrélations & Habitat", "COVID, Biais", "Séries temporelles des observations", "Synthèse & Recommandations"]
st.sidebar.markdown("---")
page = st.sidebar.selectbox("Sélectionnez une page", pages)

st.sidebar.title("Filtres Interactifs")

# Filtres
annees = st.sidebar.multiselect("Années", options=sorted(df['annee'].unique()), default=[2018, 2019, 2020, 2021, 2022])
saisons = st.sidebar.multiselect("Saisons", options=['Printemps', 'Été', 'Automne', 'Hiver'], default=['Printemps', 'Été'])
etiquettes = st.sidebar.multiselect("Groupes taxonomiques", options=df['etiquette'].unique(), default=['oiseau'])
departements = st.sidebar.multiselect("Départements", options=df['departement'].unique())

# Appliquer filtres
df_filtered = df[
    (df['annee'].isin(annees)) &
    (df['saison'].isin(saisons)) &
    (df['etiquette'].isin(etiquettes))
]
if departements:
    df_filtered = df_filtered[df_filtered['departement'].isin(departements)]

# === PAGE ACCUEIL ===
if page == "Accueil & Storytelling":
# if st.sidebar.button("Accueil & Storytelling"):
    st.title("Biodiversité ou Observateurs ?")
    st.markdown("### *Une cartographie honnête de l’effort citoyen dans le Grand Est (2010–2024)*")
    
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("""
        <div class="highlight">
        <strong>«En 2020, la nature n’a pas explosé…  
         c’est nous qui l’avons observée comme jamais.»</strong>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
                    # **Biodiversité ou Observateurs ?**  
*Ce que les données du Grand Est nous disent vraiment (2010–2024)*

---

## **1. L’Année où la Nature a « Explosé »**  
> **2020 : +107 % d’observations**  
> *Mais pas une seule espèce n’a migré en masse.*  

**LOESS lissé** : le pic est **isolé**.  
**Réalité** : confinement → plus de temps, plus de smartphones, plus de passionnés dans les jardins.  
**Pas d’effet écologique.**  

---

## **2. La Biodiversité, Sport de Village**  
> **Bergholtz (1 100 hab.) → 402 obs/km²**  
> **Strasbourg → < 10 obs/km²**  

**Heatmaps** : les points rouges **ne suivent pas les forêts**, mais les **villages**.  
**Top 10 hotspots** : petites communes, grands passionnés.  

---

## **3. Même les Plantes des Marais… sont en Ville**  
> **Heatmap « plantes humides »** : mêmes hotspots que la carte totale.  
> **Corrélation habitat naturel → observations = 0.05 (non significative)**  

**Conclusion** : on observe là où on vit.  
**Pas là où la nature est.**

---

## **4. Pas de Déclin Prouvé, Juste un Manque de Regard**  
> **Avant 2010 → 0 donnée**  
> **Espèces réglementées → même pic 2020**  
> **Zones blanches → réserves naturelles invisibles**  

**Absence de données ≠ absence de vie.**

---

## **5. Et Si On Élargissait le Regard ?**  
> **Proposition** :  
> - Équiper **chaque maire** d’un smartphone  
> - Former **les écoles, EHPAD, mairies**  
> - Créer **1 sentinelle biodiversité par village**  

**Objectif** : transformer **chaque habitant en capteur vivant**.  
**Pour que demain, la carte reflète enfin la nature… et pas seulement nos passions.**

---

> **Ce n’est pas une carte de la biodiversité.**  
> **C’est une carte de notre regard.**  
> **Et si on l’ouvrait à tous ?**

---
*Concours DataGrandEst 2025 – Thème : Biodiversité*  
*Données : Faune-GrandEst, INPN, citoyens naturalistes*
                    """)
        
    with col2:
        st.image("https://m.espacepourlavie.ca/blogue/sites/espacepourlavie.ca.blogue/files/styles/album-650x435/public/ornithoptera_priamus_poseidon_femelle_australie.jpg?itok=tdj1p6em", use_column_width=True)


# === PAGE : ANALYSES TEMPORELLES ===
elif page == "Analyses Temporelles":
# elif st.sidebar.button("Analyses Temporelles"):
    st.title("Analyses Temporelles")
    
    tab1, tab2, tab3 = st.tabs(["Évolution Totale", "Par Groupe", "COVID & LOESS"])
    
    with tab1:
        obs_ann = df_filtered.groupby('annee').size().reset_index(name='obs')
        fig = px.line(obs_ann, x='annee', y='obs', markers=True, title="Évolution des Observations")
        fig.update_layout(template="simple_white")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        obs_groupe = df_filtered.groupby(['annee', 'etiquette']).size().reset_index(name='obs')
        fig = px.area(obs_groupe, x='annee', y='obs', color='etiquette', title="Par Groupe Taxonomique")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        df_covid = df_filtered[df_filtered['annee'].between(2018, 2022)]
        obs_covid = df_covid.groupby('annee').size().reset_index(name='obs')
        lowess_res = lowess(obs_covid['obs'], obs_covid['annee'], frac=0.5)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=obs_covid['annee'], y=obs_covid['obs'], mode='lines+markers', name='Observations'))
        fig.add_trace(go.Scatter(x=obs_covid['annee'], y=lowess_res[:, 1], mode='lines', name='LOESS', line=dict(color='red')))
        fig.update_layout(title="Lissage LOESS autour de COVID", template="simple_white")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**Interprétation** : Le pic 2020 est une **anomalie sociale**, pas écologique.")

# === PAGE : ANALYSES SPATIALES ===
# elif st.sidebar.button("Analyses Spatiales"):
elif page == "Analyses Spatiales":
    st.title("Cartographie Interactive")
    
    # html_files = {
    #     "Heatmap Globale": "heatmap_hotspots.html",
    #     "Plantes Humides": "heatmap_plantes_humides.html",
    #     "Communes (Choroplèthe)": "carte_observations_communes.html",
    #     "Hotspots Biodiv": "hotspots_biodiv.html"
    # }
        
    # selected_map = st.selectbox("Choisir une carte", list(html_files.keys()))
    
    # with open(html_files[selected_map], "r", encoding="utf-8") as f:
    #     html_data = f.read()
    
    # st.components.v1.html(html_data, height=600)
    
    # --- Dictionnaire de tes cartes Google Drive ---
    html_files = {
        "Heatmap Globale": "https://drive.google.com/file/d/1CSL6q8L7hXXaCMUgunjCA3CIOnYzIUVz",
        "Plantes Humides": "https://drive.google.com/file/d/1bNH1kWCoaYUl_F8nhAgqj58ycWiAZ7cR/view?usp=sharing",
        "Communes (Choroplèthe)": "https://drive.google.com/file/d/1xaUlYWT74Jg4hmjo614lyn1MxlpWUSw7/view?usp=sharing",
        "Hotspots Biodiv": "https://drive.google.com/file/d/12C6i5nOUa4MY9laAdsezeQP6o0eCFWFr/view?usp=sharing"
    }

    # --- Sélecteur Streamlit ---
    selected_map = st.selectbox("Choisir une carte :", list(html_files.keys()))

    # --- Fonction utilitaire pour télécharger et charger une carte HTML ---
    @st.cache_data
    def load_map_from_drive(drive_url, filename):
        """
        Télécharge un fichier HTML depuis Google Drive (si nécessaire)
        et retourne le chemin local du fichier.
        """
        # Extraction de l’ID du lien Drive (que le lien ait /d/ ou ?id=)
        if "/d/" in drive_url:
            file_id = drive_url.split("/d/")[1].split("/")[0]
        elif "id=" in drive_url:
            file_id = drive_url.split("id=")[1].split("&")[0]
        else:
            raise ValueError("Lien Google Drive non valide.")

        url = f"https://drive.google.com/uc?id={file_id}"
        output_path = os.path.join("maps", f"{filename}.html")

        # Création du dossier local si besoin
        os.makedirs("maps", exist_ok=True)

        # Téléchargement uniquement si le fichier n'existe pas
        if not os.path.exists(output_path):
            with st.spinner(f"Téléchargement de {filename}..."):
                gdown.download(url, output_path, quiet=False)

        return output_path

    # --- Récupération du lien sélectionné ---
    selected_url = html_files[selected_map]

    # --- Téléchargement/chargement de la carte ---
    map_path = load_map_from_drive(selected_url, selected_map.replace(" ", "_"))

    # --- Lecture et affichage de la carte dans Streamlit ---
    with open(map_path, "r", encoding="utf-8") as f:
        html_data = f.read()

    st.components.v1.html(html_data, height=600, scrolling=True)
    
    if selected_map == "Communes (Choroplèthe)":
        st.markdown("""
        **• Pas de "zones blanches" → réseau dense de communes contributrices**\n
        **• Corrélation évidente avec la densité de population**\n
        **• Les villes moyennes dominent** (pas Paris, pas les métropoles lointaines)\n
        **• Biais géographique clair: plus d'observateurs = plus de points**
        """)
        st.success("**La carte des observations est la carte des habitants actifs, pas de la biodiversité.**")
    
    if selected_map == "Heatmap Globale":
        st.markdown("""
        **• Pas de hotspots en pleine nature → aucun signal dans les parcs nationaux** \n
        **• Effet "agglomération" : plus de gens = plus de données** \n
        """)
        st.success("**Il s'agit de la chaleur humaine, pas écologique.**")
    
    if selected_map == "Hotspots Biodiv":
        st.markdown("""
        **• Top 10 des "capitales de la science citoyenne"** \n
        **• Aucun hotspot en Suisse, Allemagne, Belgique → données locales uniquement** 
        """)
        st.success("**La biodiversité observée est un phénomène local, urbain, et passionné.**")
    
    if selected_map == "Plantes Humides":
        st.markdown("""
        **• Les plantes des zones humides sont observées... dans les villes!** \n
        **• Aucune corrélation avec I'habitat** \n
        **• Preuve ultime du biais d'effort**
        """)
        st.success("""**Même les espèces "naturelles" sont vues là où il y a des humains.**""")
    
    
# === PAGE : CORRÉLATIONS & HABITAT ===
# elif st.sidebar.button("Corrélations & Habitat"):
elif page == "Corrélations & Habitat":
    st.title("Corrélations Habitat")
    
    df_commune = df_filtered.groupby('codeInseeCommune').agg(
        observations=('codeInseeCommune', 'size'),
        urbain=('urbain', 'mean'),
        agricole=('agricole', 'mean'),
        naturel=('naturel', 'mean'),
        humide=('humide', 'mean')
    ).reset_index().dropna()
    
    corr = df_commune[['observations', 'urbain', 'agricole', 'naturel', 'humide']].corr()
    
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdYlGn", title="Matrice de Corrélation")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""**+0.27 avec urbain → plus de béton = plus d'observations (accès, mangeoires, parcs).**""")
    st.success("""**L'effort humain prime sur l'habitat naturel**""")
    
    st.title("Régression OLS (observations ~ surfaces)")
    st.markdown("""
                **Modèle faible (R² = 0.075) → l'habitat explique peu.**\n
                Corrélations ciblées\n
            • Oiseaux ↔ Naturel : +0.046 (p = 0.018) → **faible mais réel** \n
            • Papillons → Humide : +0.073 (p = 0.003) → **lien avec prairies humides**\n
                """)
    st.success("""
              **L'effort humain domine.**
              """)
    
# === PAGE : BIAIS & NORMALISATIONS ===
# elif st.sidebar.button("Biais & Normalisations"):
elif page == "COVID, Biais":
    st.title("pré/post-COVID")
    # 3.2 Comparer pré/post-COVID
    pre_covid = df[df['annee'] < 2020].groupby('annee').size()
    post_covid = df[df['annee'] >= 2020].groupby('annee').size()  # Inclut 2020-2021 comme dans code précédent
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Observations pré-COVID**")
        st.dataframe(pre_covid, use_container_width=True)
    with col2:
        st.markdown("**Observations post-COVID**")
        st.dataframe(post_covid, use_container_width=True)
    
    # Taux moyen pré vs. post
    moy_pre = pre_covid.mean()
    moy_post = post_covid.mean()
    st.warning(f"Taux moyen pré-COVID : {moy_pre}, post-COVID : {moy_post} (variation : {(moy_post - moy_pre)/moy_pre * 100:.2f}%)")
    
    
        # --- Conversion en DataFrame pour Plotly ---
    pre_df = pre_covid.reset_index()
    pre_df.columns = ['annee', 'observations']
    post_df = post_covid.reset_index()
    post_df.columns = ['annee', 'observations']

    # --- Création du graphique interactif ---
    fig = go.Figure()

    # Barres pré-COVID
    fig.add_trace(go.Bar(
        x=pre_df['annee'],
        y=pre_df['observations'],
        name='Pré-COVID',
        marker_color='steelblue'
    ))

    # Barres post-COVID
    fig.add_trace(go.Bar(
        x=post_df['annee'],
        y=post_df['observations'],
        name='Post-COVID',
        marker_color='orange'
    ))

    # Ligne verticale pour marquer le début du COVID (à 2020)
    fig.add_vline(
        x=2019.5,  # juste avant 2020
        line_dash='dash',
        line_color='red',
        annotation_text='Début COVID',
        annotation_position='top right'
    )

    # --- Personnalisation du layout ---
    fig.update_layout(
        title='Observations Pré / Post COVID',
        xaxis_title='Année',
        yaxis_title='Nombre d\'observations',
        barmode='group',
        template='plotly_white',
        legend=dict(title='', orientation='h', y=-0.2, x=0.3)
    )

    # --- Affichage interactif dans Streamlit ---
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("""**2020 = année charnière → confinement = boom naturaliste.**""")
    
    
    st.title("Biais d'Observation")
    effort = df_filtered.groupby(['commune', 'annee'])['obsID'].nunique().reset_index(name='effort')
    obs_total = df_filtered.groupby(['commune', 'annee']).size().reset_index(name='obs')
    df_biais = pd.merge(obs_total, effort, on=['commune', 'annee'])
    df_biais['normalise'] = df_biais['obs'] / df_biais['effort']
    
    fig = px.histogram(df_biais, x='normalise', nbins=30, title="Distribution des Observations Normalisées")
    st.plotly_chart(fig, use_container_width=True)
    
    top_communes = df_biais.sort_values('normalise', ascending=False)
    st.dataframe(top_communes[['commune', 'annee', 'normalise']], use_container_width=True)
    
    st.success("""**Quelques communes ultra-actives portent 80% des données.**""")


elif page == "Séries temporelles des observations":
    # Exemple : afficher en top de la page
    st.title("Séries temporelles des observations")

    # ---------- Pré-traitement ----------
    df = df.copy()

    # Assurer que 'annee' est entier
    if 'annee' not in df.columns or 'mois' not in df.columns:
        st.error("Les colonnes 'annee' et 'mois' sont requises dans le dataset.")
        st.stop()

    df['annee'] = pd.to_numeric(df['annee'], errors='coerce')
    df = df.dropna(subset=['annee'])
    df['annee'] = df['annee'].astype(int)

    # Convertir 'mois' en numéro de mois si besoin
    if df['mois'].dtype == object:
        # tente plusieurs parseurs
        df['mois_num'] = pd.to_datetime(df['mois'], format='%B', errors='coerce').dt.month
        if df['mois_num'].isna().all():
            df['mois_num'] = pd.to_datetime(df['mois'], errors='coerce').dt.month
    else:
        df['mois_num'] = pd.to_numeric(df['mois'], errors='coerce')

    df = df.dropna(subset=['mois_num'])
    df['mois_num'] = df['mois_num'].astype(int)

    # Option : choix d'une espèce / catégorie (si colonne 'espece' existe)
    group_by_col = None
    if 'espece' in df.columns:
        if st.checkbox("Grouper par espèce (affiche une espèce à la fois)", value=False):
            group_by_col = 'espece'
            unique_species = sorted(df['espece'].dropna().unique())
            selected_species = st.selectbox("Sélectionner une espèce", unique_species)
            df = df[df['espece'] == selected_species]

    # ---------- Calculs pour les graphiques ----------
    obs_annuelles = df.groupby('annee').size().reset_index(name='observations').sort_values('annee')
    obs_mensuelles = df.groupby('mois_num').size().reset_index(name='observations').sort_values('mois_num')

    # Ajouter label mois en clair
    month_labels = {m: calendar.month_name[m] for m in range(1, 13)}
    obs_mensuelles['mois_label'] = obs_mensuelles['mois_num'].map(month_labels)

    # ---------- Interface onglets ----------
    tab1, tab2, tab3 = st.tabs(["Évolution annuelle", "LOESS smoothing", "Évolution mensuelle"])

    with tab1:
        st.subheader("Évolution Annuelle des Observations")
        if obs_annuelles.empty:
            st.info("Pas de données annuelles disponibles.")
        else:
            fig_ann = px.line(obs_annuelles, x='annee', y='observations', markers=True,
                            title="Évolution Annuelle des Observations Totales")
            # Ajout d'un trace supplémentaire pour rendre légende explicite (si besoin)
            fig_ann.update_traces(name="Observations", selector=dict(mode='markers+lines'))
            fig_ann.update_layout(legend=dict(title="Légende"))
            st.plotly_chart(fig_ann, use_container_width=True)
        st.success("""**Les observations ont connu une forte hausse jusqu’en 2020, année du pic maximal, avant de chuter brutalement après 2022, suggérant une interruption ou une baisse importante de la collecte des données.**""")
        
    with tab2:
        st.subheader("Évolution Annuelle avec LOESS Smoothing")
        if obs_annuelles.shape[0] < 3:
            st.info("Pas assez de points pour LOESS (au moins 3 années requises).")
        else:
            x = obs_annuelles['annee'].values
            y = obs_annuelles['observations'].values
            frac = st.slider("Paramètre LOESS `frac` (partie locale utilisée)", min_value=0.05, max_value=0.8, value=0.2, step=0.05)
            lowess_res = lowess(y, x, frac=frac)
            # Construire figure Plotly avec deux traces (observations + smoothing)
            fig_loess = go.Figure()
            fig_loess.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Observations'))
            fig_loess.add_trace(go.Scatter(x=lowess_res[:, 0], y=lowess_res[:, 1], mode='lines', name=f'LOESS (frac={frac})',
                                        line=dict(dash='dash')))
            fig_loess.update_layout(title="Évolution Annuelle avec LOESS",
                                    xaxis_title="Année", yaxis_title="Nombre d'observations",
                                    legend=dict(title="Légende"))
            st.plotly_chart(fig_loess, use_container_width=True)
        st.success("""**LOESS smoothing : confirme tendance haussière pré-COVID, palier post, pas de déclin structurel.**""")
    
    with tab3:
        st.subheader("Évolution Mensuelle (agrégée sur toutes les années)")
        if obs_mensuelles.empty:
            st.info("Pas de données mensuelles disponibles.")
        else:
            # Ordre garanti par mois_num
            fig_month = go.Figure()
            fig_month.add_trace(go.Scatter(
                x=obs_mensuelles['mois_num'],
                y=obs_mensuelles['observations'],
                mode='lines+markers',
                name='Observations agrégées'
            ))
            # Remplacer ticks par noms de mois
            fig_month.update_layout(
                title="Évolution Mensuelle des Observations (Agrégée sur toutes les années)",
                xaxis=dict(
                    tickmode='array',
                    tickvals=obs_mensuelles['mois_num'],
                    ticktext=obs_mensuelles['mois_label'],
                    title='Mois'
                ),
                yaxis=dict(title='Nombre d\'observations'),
                legend=dict(title="Légende")
            )
            st.plotly_chart(fig_month, use_container_width=True)
        st.success("""**Saisonnalité hivernale → nourrissage, visibilité.**""")

# === PAGE : INTERPRÉTATIONS FINALES ===
# elif st.sidebar.button("Synthèse & Recommandations"):
elif page == "Synthèse & Recommandations":
    st.title("Synthèse Finale")
    
    st.markdown("""
    <div class="highlight">
    <h3>Ce que les données disent vraiment :</h3>
    <ul>
        <li><strong>2020 ≠ explosion écologique</strong> → +107 % d'observations = confinement</li>
        <li><strong>La biodiversité observée = carte des passionnés</strong>, pas de la nature</li>
        <li><strong>Zones blanches ≠ absence de vie</strong> → absence d'observateurs</li>
        <li><strong>Pas de preuve de déclin</strong> → manque de données, pas de données historiques</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.image("https://www.terre-sauvage.com/echos-de-la-nature/operationpapillons-c-viginature.jpg/image_preview", caption="Opération Papillons – Exemple de mobilisation citoyenne")
    
    st.markdown("### **Recommandations**")
    st.markdown("""
    1. **Mobiliser les mairies et écoles** dans les zones blanches  
    2. **Former les agents publics** à la saisie (Faune-GrandEst)  
    3. **Créer un réseau de "sentinelles biodiversité"** par village  
    4. **Normaliser systématiquement par effort** dans les rapports
    """)

# === FOOTER ===
st.sidebar.markdown("---")
st.sidebar.markdown("**Concours DataGrandEst 2025** – Thème : *Biodiversité*")
st.sidebar.markdown("Made by Codjo Ulrich Expéra AKAKPO")




