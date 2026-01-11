import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import plotly.express as px


# App config
st.set_page_config(page_title="California Housing – Predictor + EDA", layout="wide")


# Data + model helpers
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("https://housing-raw.s3.amazonaws.com/california_housing.csv")
    # for transformed data: I use the unmodified here because I am poor
    # "https://housing-500.s3.amazonaws.com/final_housing_500.csv"

    df["ocean_proximity"] = df["ocean_proximity"].replace(
        {"NEAR BAY": "NEAR WATER", "NEAR OCEAN": "NEAR WATER", "ISLAND": "NEAR WATER"}
    )

    df = df.dropna().copy()
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    y = df["median_house_value"]
    X = df.drop(columns=["median_house_value"])

    categorical_features = ["ocean_proximity"]
    numeric_features = [c for c in X.columns if c not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(drop="first"), categorical_features),
        ]
    )

    model = GradientBoostingRegressor(
        n_estimators=800,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.85,
        random_state=42,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    return pipeline, r2, rmse


def nearest_weather(df: pd.DataFrame, lat: float, lon: float, weather_cols: list[str]) -> dict:
    coords = df[["latitude", "longitude"]].to_numpy(dtype=float)
    target = np.array([lat, lon], dtype=float)
    d2 = np.sum((coords - target) ** 2, axis=1)
    idx = int(np.argmin(d2))
    row = df.iloc[idx]
    return {c: float(row[c]) for c in weather_cols}


def build_user_input(df: pd.DataFrame, auto_weather: bool) -> pd.DataFrame:
    st.subheader("Predict House Price")

    def slider(col, fmt="%.3f"):
        return st.slider(
            col,
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].median()),
            format=fmt,
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        longitude = slider("longitude", fmt="%.4f")
        latitude = slider("latitude", fmt="%.4f")
        housing_median_age = slider("housing_median_age", fmt="%.0f")

    with col2:
        total_rooms = slider("total_rooms", fmt="%.0f")
        total_bedrooms = slider("total_bedrooms", fmt="%.0f")
        population = slider("population", fmt="%.0f")

    with col3:
        households = slider("households", fmt="%.0f")
        median_income = slider("median_income", fmt="%.3f")
        ocean_opts = sorted(df["ocean_proximity"].unique().tolist())
        ocean_proximity = st.selectbox("ocean_proximity", ocean_opts, index=0)

    weather_cols = [c for c in ["temperature_2m_mean", "sunshine_duration", "precipitation_sum"] if c in df.columns]

    if weather_cols:
        st.subheader("Weather Features (from ETL)")
        w1, w2, w3 = st.columns(3)

        if auto_weather:
            wvals = nearest_weather(df, latitude, longitude, weather_cols)
            temperature_2m_mean_default = wvals.get("temperature_2m_mean", float(df["temperature_2m_mean"].median())) if "temperature_2m_mean" in weather_cols else None
            sunshine_duration_default = wvals.get("sunshine_duration", float(df["sunshine_duration"].median())) if "sunshine_duration" in weather_cols else None
            precipitation_sum_default = wvals.get("precipitation_sum", float(df["precipitation_sum"].median())) if "precipitation_sum" in weather_cols else None
        else:
            temperature_2m_mean_default = float(df["temperature_2m_mean"].median()) if "temperature_2m_mean" in weather_cols else None
            sunshine_duration_default = float(df["sunshine_duration"].median()) if "sunshine_duration" in weather_cols else None
            precipitation_sum_default = float(df["precipitation_sum"].median()) if "precipitation_sum" in weather_cols else None

        temperature_2m_mean = None
        sunshine_duration = None
        precipitation_sum = None

        if "temperature_2m_mean" in weather_cols:
            with w1:
                temperature_2m_mean = st.slider(
                    "temperature_2m_mean",
                    float(df["temperature_2m_mean"].min()),
                    float(df["temperature_2m_mean"].max()),
                    float(temperature_2m_mean_default),
                    format="%.2f",
                )
        if "sunshine_duration" in weather_cols:
            with w2:
                sunshine_duration = st.slider(
                    "sunshine_duration",
                    float(df["sunshine_duration"].min()),
                    float(df["sunshine_duration"].max()),
                    float(sunshine_duration_default),
                    format="%.0f",
                )
        if "precipitation_sum" in weather_cols:
            with w3:
                precipitation_sum = st.slider(
                    "precipitation_sum",
                    float(df["precipitation_sum"].min()),
                    float(df["precipitation_sum"].max()),
                    float(precipitation_sum_default),
                    format="%.1f",
                )

    user_row = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }

    if "temperature_2m_mean" in df.columns:
        user_row["temperature_2m_mean"] = float(temperature_2m_mean) if temperature_2m_mean is not None else float(df["temperature_2m_mean"].median())
    if "sunshine_duration" in df.columns:
        user_row["sunshine_duration"] = float(sunshine_duration) if sunshine_duration is not None else float(df["sunshine_duration"].median())
    if "precipitation_sum" in df.columns:
        user_row["precipitation_sum"] = float(precipitation_sum) if precipitation_sum is not None else float(df["precipitation_sum"].median())

    user_df = pd.DataFrame([user_row])
    return user_df


# UI
st.title("California Housing – Dashboard + Price Predictor")

with st.sidebar:
    st.header("Settings")

    # (REMOVED) Plot sample size feature

    show_raw = st.checkbox("Show raw data table", value=False)

    st.subheader("Map settings")
    map_point_size = st.slider("Point size scale", 2, 12, 10)

    st.subheader("Weather settings")
    auto_weather = st.checkbox("Auto-fill weather from nearest location", value=True)


df = load_data()
pipeline, r2, rmse = train_model(df)

m1, m2, m3 = st.columns(3)
m1.metric("Model", "Gradient Boosting")
m2.metric("Test R²", f"{r2:.3f}")
m3.metric("Test RMSE", f"{rmse:,.0f} USD")


# Prediction section
user_row = build_user_input(df, auto_weather=auto_weather)
pred_single = float(pipeline.predict(user_row)[0])

st.markdown("### Predicted Price")
st.success(f"Estimated median house value: **{pred_single:,.0f} USD**")

with st.expander("Show the exact input row used for prediction"):
    st.dataframe(user_row)

if show_raw:
    st.subheader("Raw data (preview)")
    st.dataframe(df.head(50))

st.divider()


# Prepare sample + predictions for plots
st.header("Exploratory Plots + Predictions")

# Always cap plots at 500 rows (no sidebar control)
plot_df = df.sample(n=min(500, len(df)), random_state=42).copy()

# Add predictions for hover residuals
X_plot = plot_df.drop(columns=["median_house_value"])
plot_df["predicted_price"] = pipeline.predict(X_plot)
plot_df["actual_price"] = plot_df["median_house_value"]
plot_df["residual"] = plot_df["actual_price"] - plot_df["predicted_price"]

hover_fields = {
    "actual_price": ":,.0f",
    "predicted_price": ":,.0f",
    "residual": ":,.0f",
    "housing_median_age": True,
    "total_rooms": True,
    "total_bedrooms": True,
    "population": True,
    "households": True,
    "median_income": ":.3f",
}

if "temperature_2m_mean" in plot_df.columns:
    hover_fields["temperature_2m_mean"] = ":.2f"
if "sunshine_duration" in plot_df.columns:
    hover_fields["sunshine_duration"] = ":,.0f"
if "precipitation_sum" in plot_df.columns:
    hover_fields["precipitation_sum"] = ":.1f"


# 1) California Map
st.subheader("1) California Map (colored by Ocean Proximity)")

map_center = {"lat": 36.5, "lon": -119.5}

fig_map = px.scatter_mapbox(
    plot_df,
    lat="latitude",
    lon="longitude",
    color="ocean_proximity",
    zoom=4.3,
    center=map_center,
    opacity=0.65,
    size="predicted_price",
    size_max=map_point_size,
    hover_name="ocean_proximity",
    hover_data=hover_fields,
)

fig_map.update_layout(
    mapbox_style="open-street-map",
    height=560,
    margin=dict(l=0, r=0, t=40, b=0)
)
st.plotly_chart(fig_map, use_container_width=True)


# 2) Income vs price
st.subheader("2) Median Income vs House Value")

fig_scatter = px.scatter(
    plot_df,
    x="median_income",
    y="actual_price",
    color="ocean_proximity",
    opacity=0.55,
    hover_data=hover_fields,
    labels={"actual_price": "median_house_value (actual)"},
)
fig_scatter.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig_scatter, use_container_width=True)


# Weather plots
if "temperature_2m_mean" in plot_df.columns:
    st.subheader("2b) Temperature vs House Value")
    fig_temp = px.scatter(
        plot_df,
        x="temperature_2m_mean",
        y="actual_price",
        color="ocean_proximity",
        opacity=0.55,
        hover_data=hover_fields,
        labels={"actual_price": "median_house_value (actual)"},
    )
    fig_temp.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_temp, use_container_width=True)

if "sunshine_duration" in plot_df.columns:
    st.subheader("2c) Sunshine Duration vs House Value")
    fig_sun = px.scatter(
        plot_df,
        x="sunshine_duration",
        y="actual_price",
        color="ocean_proximity",
        opacity=0.55,
        hover_data=hover_fields,
        labels={"actual_price": "median_house_value (actual)"},
    )
    fig_sun.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_sun, use_container_width=True)

if "precipitation_sum" in plot_df.columns:
    st.subheader("2d) Precipitation Sum vs House Value")
    fig_prec = px.scatter(
        plot_df,
        x="precipitation_sum",
        y="actual_price",
        color="ocean_proximity",
        opacity=0.55,
        hover_data=hover_fields,
        labels={"actual_price": "median_house_value (actual)"},
    )
    fig_prec.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_prec, use_container_width=True)


# 3) Price distribution histogram (Actual) by ocean_proximity
st.subheader("3) Actual Price Distribution (Stacked by Ocean Proximity)")

bins = st.slider("Histogram bins", 10, 80, 35, step=5)
fig_hist = px.histogram(
    plot_df,
    x="actual_price",
    color="ocean_proximity",
    nbins=bins,
    barmode="stack",
    opacity=0.85,
    labels={"actual_price": "median_house_value (actual)"},
)
fig_hist.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig_hist, use_container_width=True)


# 4) Boxplot: actual price by ocean proximity
st.subheader("4) Actual House Value by Ocean Proximity (Boxplot)")

fig_box = px.box(
    plot_df,
    x="ocean_proximity",
    y="actual_price",
    points="outliers",
    labels={"actual_price": "median_house_value (actual)"},
)
fig_box.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig_box, use_container_width=True)

st.divider()
st.caption(
    "Weather features are included in the model if they exist in the ETL output (temperature_2m_mean, sunshine_duration, precipitation_sum). "
    "If auto-fill is enabled, weather inputs are taken from the nearest location in the dataset based on latitude/longitude."
)