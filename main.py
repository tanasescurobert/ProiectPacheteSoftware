import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import numpy as np
import branca

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("dataset_2023.csv")
    return df

data = load_data()

st.title("PIB per locuitor, rata somajului si rata de ocupare a populatiei in varsta de munca (15-64 ani) in Europa si Asia Centrala")

# Tabs
tabs = st.tabs([
    "1) Prelucrare & Statistici",
    "2) Outlieri",
    "3) Harta interactiva",
    "4) Codificare & Scalare",
    "5) Clusterizare",
    "6) Regresie Multiplă",
    "7) Clasificare"
])





# ---------------------- TAB 1 ----------------------
with tabs[0]:
    st.header("Gruparea datelor si tratarea valorilor lipsa")
    grouped_data = data.groupby("Indicator")["Value"].mean()
    st.write("Media indicatorilor")
    st.dataframe(grouped_data)

    for indicator in data['Indicator'].unique():
        indicator_data = data[data['Indicator'] == indicator]
        indicator_mean = indicator_data['Value'].mean()
        data.loc[data['Indicator'] == indicator, 'Value'] = data.loc[data['Indicator'] == indicator, 'Value'].fillna(indicator_mean)

    st.write("Setul de date cu valorile lipsa inlocuite cu media pe indicator")
    st.dataframe(data)

    data_pivot = data.pivot_table(index='Country Name', columns='Indicator', values='Value', aggfunc='mean')
    st.write("Gruparea setului de date dupa tara")
    st.dataframe(data_pivot)





# ---------------------- TAB 2 ----------------------
with tabs[1]:
    st.header("Detectarea valorilor extreme - GDP per capita")
    indicator_name = 'GDP per capita'
    indicator_data = data[data['Indicator'] == indicator_name]
    indicator_mean = indicator_data["Value"].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(y=indicator_data["Value"], ax=ax)

    Q1 = indicator_data["Value"].quantile(0.25)
    Q3 = indicator_data["Value"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = indicator_data[(indicator_data["Value"] < lower_bound) | (indicator_data["Value"] > upper_bound)]

    for i, row in indicator_data.iterrows():
        ax.text(x=0, y=row["Value"], s=row["Country Name"], ha="center", va="bottom", fontsize=9, color="blue")

    st.pyplot(fig)





# ---------------------- TAB 3 ----------------------
with tabs[2]:
    st.header("Harta interactiva - GDP per capita")
    gdf = gpd.read_file("countries/ne_110m_admin_0_countries.shp")
    countries = data['Country Name'].unique()
    gdf_filtered = gdf[gdf['NAME'].isin(countries)]
    data_map = gdf_filtered.merge(data, left_on="NAME", right_on="Country Name", how="left")

    m = folium.Map(location=[50, 10], zoom_start=4)
    indicator_data = data[data['Indicator'] == 'GDP per capita']
    indicator_mean = indicator_data['Value'].mean()

    def get_color(value):
        return 'red' if value < indicator_mean else 'green' if value > indicator_mean else 'gray'

    for _, row in data_map[data_map['Indicator'] == 'GDP per capita'].iterrows():
        country_name = row["Country Name"]
        value = row["Value"]
        tooltip_text = f"{country_name}<br>Indicator: {row['Indicator']}<br>Value: {value:.2f}"

        folium.GeoJson(
            row["geometry"],
            tooltip=folium.Tooltip(tooltip_text),
            style_function=lambda feature, val=value: {
                'fillColor': get_color(val),
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.3
            }
        ).add_to(m)

    st.write("Harta in functie de media PIB-ului pe cap de locuitor:")
    st.markdown("""
        <div style='display: flex; gap: 10px'>
            <p style='color: green'>Verde -> peste medie</p> /
            <p style='color: red;'>Rosu -> sub medie</p> /
            <p style='color: gray;'>Gri -> valoare lipsa (inlocuita cu media)</p>
        </div>
    """, unsafe_allow_html=True)

    folium_static(m)





# ---------------------- TAB 4 ----------------------
with tabs[3]:
    st.header("Codificare si scalare")
    df_encoded = data.copy()
    le = LabelEncoder()
    df_encoded['Country_Code'] = le.fit_transform(df_encoded['Country Name'])
    st.write("Exemplu codificare label:", df_encoded[['Country Name', 'Country_Code']].drop_duplicates().head())

    # Scalare
    pivot_scaled = data_pivot.fillna(data_pivot.mean())
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot_scaled)
    st.write("Date scalate:")
    st.dataframe(pd.DataFrame(scaled_data, index=pivot_scaled.index, columns=pivot_scaled.columns))





# ---------------------- TAB 5 ----------------------
with tabs[4]:
    st.header("Clusterizare (KMeans)")
    kmeans = KMeans(n_clusters=3, random_state=0)
    clusters = kmeans.fit_predict(scaled_data)

    st.write("Etichetare tari pe clustere:")
    cluster_df = pd.DataFrame({"Country": pivot_scaled.index, "Cluster": clusters})
    st.dataframe(cluster_df)





# ---------------------- TAB 6 ----------------------
with tabs[5]:
    st.header("Regresie multiplă")

    # Ensure the necessary columns are present
    required_columns = ["Unemployment rate", "Participation rate", "GDP per capita"]

    # Filter out rows where any required column has missing values
    df_reg = data_pivot[required_columns].dropna()

    # Check if all required columns are available
    if all(col in df_reg.columns for col in required_columns):
        # Define X (independent variables) and y (dependent variable)
        X = df_reg[["Unemployment rate", "Participation rate"]]
        y = df_reg["GDP per capita"]

        # Add a constant to the independent variables (for the intercept term)
        X = sm.add_constant(X)

        # Perform the regression
        model = sm.OLS(y, X).fit()

        # Display the regression results
        st.write(model.summary())
    else:
        st.warning("Nu toate coloanele necesare pentru regresia multiplă sunt prezente în setul de date.")





# ---------------------- TAB 7 ----------------------
with tabs[6]:
    st.header("Clasificare - Logistic Regression")
    df_clf = data_pivot[["Unemployment rate", "GDP per capita"]].dropna()
    df_clf["Above Median"] = (df_clf["GDP per capita"] > df_clf["GDP per capita"].median()).astype(int)

    X = df_clf[["Unemployment rate"]]
    y = df_clf["Above Median"]

    clf = LogisticRegression()
    clf.fit(X, y)

    st.write("Coeficienti:", clf.coef_)
    st.write("Intercept:", clf.intercept_)

    st.write("Predicție pentru valori noi:")
    example_value = st.slider("Rata somajului (exemplu):", float(X.min()), float(X.max()), float(X.mean()))
    prediction = clf.predict([[example_value]])[0]
    st.write(f"Clasificare: {'Peste medie' if prediction == 1 else 'Sub medie'}")
