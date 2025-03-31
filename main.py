import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import branca


def load_data():
    df = pd.read_csv("dataset_2023.csv")
    return df

data = load_data()

st.markdown("<h1 style='color: yellow; border: 1px solid; border-color: red;'>PIB per locuitor, rata somajului si rata de ocupare a populatiei in varsta de munca (15-64 ani) in Europa si Asia Centrala</h1>"
            , unsafe_allow_html=True)

st.header("Gruparea datelor si tratarea valorilor lipsa")
st.write("Media indicatorilor")
grouped_data = data.groupby("Indicator")["Value"].mean()
st.write(grouped_data)

for indicator in data['Indicator'].unique():
    indicator_data = data[data['Indicator'] == indicator]
    indicator_mean = indicator_data['Value'].mean()
    data.loc[data['Indicator'] == indicator, 'Value'] = data.loc[data['Indicator'] == indicator, 'Value'].fillna(indicator_mean)

st.write("Inlocuirea valorilor lipsa in setul de date")
st.write(data)

data_pivot = data.pivot_table(index='Country Name', columns='Indicator', values='Value', aggfunc='mean')
st.write("Gruparea setului de date dupa tara")
st.write(data_pivot)





indicator_name = 'GDP per capita'
indicator_data = data[data['Indicator'] == indicator_name]
indicator_mean = indicator_data["Value"].mean()


st.header("Date despre PIB pe cap de locuitor")


# Boxplot
st.write("Boxplot valori extreme")

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


# Harta interactiva
gdf = gpd.read_file("countries/ne_110m_admin_0_countries.shp")
countries = data['Country Name'].unique()
gdf_filtered = gdf[gdf['NAME'].isin(countries)]
data_map = gdf_filtered.merge(data, left_on="NAME", right_on="Country Name", how="left")

m = folium.Map(location=[50, 10], zoom_start=4)  # Europa centrala

colormap = branca.colormap.LinearColormap(
    colors=['green', 'yellow', 'red'],
    vmin=indicator_data["Value"].min(),
    vmax=indicator_data["Value"].max()
).to_step(n=6)

st.write("Date indicator:")
st.write(indicator_data["Value"].describe())

st.write("Harta in functie de media PIB-ului pe cap de locuitor:")
st.markdown("<div style='display: flex; gap: 10px'><p style='color: green'>Verde -> peste medie</p> / <p style='color: red;'>Rosu -> sub medie</p> / <p style='color: gray;'>Gri -> valoare lipsa, a fost inlocuita cu media</p></div>", unsafe_allow_html=True)
def get_color(value):
    return 'red' if value < indicator_mean else 'green' if value > indicator_mean else 'gray'


for _, row in  data_map[data_map['Indicator'] == indicator_name].iterrows():
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

folium_static(m)
