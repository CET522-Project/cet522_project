# seattle_geo = load_geo_data("https://drive.google.com/uc?export=view&id=172kqatuR-BAc9LqpkkpFq8o7DB1bO3t0")
# st.write('## Seattle Roads', seattle_geo.plot())

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import matplotlib.colors as mcolors
import branca.colormap as cm


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Micromobility Explorer",
    page_icon="🛴",
    layout="wide",
)

st.title("Micromobility Explorer 🛴 🗺️ 🔭")
st.caption("Explore micromobility in Seattle and Spokane!")

# ----------------------------
# Data loading
# ----------------------------
@st.cache_data
def load_data_from_path(path):
    df = pd.read_csv(path)
    return  (df)

@st.cache_data
def load_geodata_from_path(path):
    gdf = gpd.read_file(path)
    return  (gdf)

CENSUS_DATA = load_geodata_from_path("census_data.geojson") # includes count data already
GRID_DATA = load_geodata_from_path("grid_data.geojson") # includes count data already
seattle_micro_streets = load_geodata_from_path("seattle-routes-data-for-all-vehicles-in-all-time.geojson").dropna(subset=["count"])
spokane_micro_streets = load_geodata_from_path("spokane-routes-data-for-all-vehicles-in-all-time.geojson").dropna(subset=["count"])

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Filters")

all_cities = ("Seattle", "Spokane")
agg_city = st.sidebar.multiselect(
    "Select cities:",
    options=all_cities,
    default=all_cities,
)
agg_map = st.sidebar.selectbox("Analysis Unit:", ["Census", "Grid",], index=0)

# ----------------------------
# Top info + KPI cards
# ----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Seattle, Average Micromobility Count", f"${seattle_micro_streets["count"].mean():,.0f}")
col2.metric("Spokane, Average Micromobility Count", f"${spokane_micro_streets["count"].mean():,.0f}")

king_county_tract_num = (CENSUS_DATA["COUNTYFP"]=="033").sum()
spokane_county_track_num = (CENSUS_DATA["COUNTYFP"]=="063").sum()
col3.metric("KING COUNTY, Number of Census Tracks", f"{king_county_tract_num:,.0f}")
col4.metric("SPOKANE COUNTY, Number of Census Tracks", f"{spokane_county_track_num:,.0f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("City of Seattle, Median Income", "$123,860")
col6.metric("City of Spokane, Median Income", "$86,206")

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["Maps", "ML Model", "Data and Summary",])

# ----------------------------
# Tab 1
# ----------------------------
with tab1:
    # make a map from variables
    def make_map_from(value, title, df):
        minx, miny, maxx, maxy = df.total_bounds
        m = folium.Map(location=[((maxy+miny)/2), ((maxx+minx)/2)], zoom_start=9)
        m.fit_bounds([[miny, minx], [maxy, maxx]]) 
        
        folium.Choropleth(
            geo_data=df.dropna(subset=value),
            data=df,
            columns=[df.index, value],
            key_on="feature.id",
            fill_color="YlGnBu",
            fill_opacity=0.7,
            line_opacity=0.2,
            #legend_name=title TODO add title to all that revence this?
        ).add_to(m)

        st_folium(m, width=600, height=550, key=str(df.count())+"_map")

    # make a line map from variables
    def make_line_map_from(value, title, df):
        minx, miny, maxx, maxy = df.total_bounds
        m = folium.Map(location=[((maxy+miny)/2), ((maxx+minx)/2)], zoom_start=9)
        m.fit_bounds([[miny, minx], [maxy, maxx]]) 

        cmap = cm.LinearColormap(
            colors=["#440154", "#31688e", "#35b779", "#fde725"],
            vmin=df[value].min(), vmax=df[value].max()
        )
        folium.GeoJson(
            df.to_crs(4326),
            style_function=lambda f: {
                "color": cmap(f["properties"][value]),
                "weight": 4,
                "opacity": 0.9
            }
        ).add_to(m)
        cmap.add_to(m)


        st_folium(m, width=600, height=550, key=str(df.count())+"_map")
        
    
    # gets tract id from city name
    def get_city_id(city_name):
        match city_name:
            case "Seattle":
                return "033"
            case "Spokane":
                return "063"
            case _:
                return "no id implemented"
    st.subheader("Let's visualize data with maps!")

    # ----------------------------
    # MAPS OF the street data
    # ----------------------------
    #TODO ADD SLIDE for minimum count value
    #TODO ADD switch for log values
    if ("Seattle" in agg_city):
        max = seattle_micro_streets["count"].max()
        make_line_map_from("count", "title", seattle_micro_streets[seattle_micro_streets["count"]>max/10])
    if ("Spokane" in agg_city):
        max = spokane_micro_streets["count"].max()
        make_line_map_from("count", "title", spokane_micro_streets[spokane_micro_streets["count"]>max/10])


    # ----------------------------
    # MAPS OF the census/grid data
    # ----------------------------

    # if a city is selected:
    if (agg_city):
        # selects data frame
        if (agg_map == "Grid"):
            data = GRID_DATA
        else:
            data = CENSUS_DATA
        
        # allow the user to pick a variable
        available_variables = ["max_count", "avg_count", "log_max_count", "log_avg_count", "POP_DENSITY_aw", "log_POP_DENSITY_aw","POP_DENSITY", "log_POP_DENSITY"]
        available_variables = list(set(available_variables) & set(data.columns))
        agg_variable = st.selectbox(
            "Which variable do you want to inspect?",
            available_variables
        )
        # maps each city selected
        for city in agg_city:

            filtered_data = data[data["COUNTYFP"]==get_city_id(city)].dropna(subset=agg_variable)
            make_map_from(agg_variable, "title", filtered_data)
                
    else:
        st.write("<- No city is selected for analysis! Select one (or more) in the sidebar to the left! ")

# ----------------------------
# Tab 2
# ----------------------------
with tab2:
    st.subheader("Lets see that ML Model!")
    # TODO by Otto

# ----------------------------
# Tab 3
# ----------------------------
with tab3:
    st.subheader("Here's a summary!")
    # TODO by Deegan

    st.write(CENSUS_DATA.head())