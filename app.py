import streamlit as st
import pandas as pd
from model import CITIES, get_live_data, train_aqi_model

st.set_page_config(page_title="Real-Time Weather & AQI Predictor", layout="wide")
st.title("🌤️ Real-Time Indian Cities Weather & AQI Predictor (AI/ML)")

st.sidebar.header("📍 Location Settings")
selected_city = st.sidebar.selectbox("Apna Shehar (City) Chunein:", list(CITIES.keys()))

@st.cache_data(ttl=600)
def load_city_data(city):
    return get_live_data(city)

with st.spinner(f"{selected_city} ka live data fetch ho raha hai..."):
    df = load_city_data(selected_city)
    model, mae, features = train_aqi_model(df)

st.subheader(f"📊 {selected_city} Ka Live Mausam Aur Pollution Status")
latest = df.iloc[-1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("🌡️ Temperature", f"{latest['temperature_2m']} °C")
col2.metric("💧 Humidity", f"{latest['relative_humidity_2m']} %")
col3.metric("💨 Wind Speed", f"{latest['wind_speed_10m']} km/h")
col4.metric("😷 Live US AQI", f"{int(latest['us_aqi'])}")

st.divider()

st.subheader("🤖 Machine Learning AQI Simulator")
st.caption(f"Random Forest Model ({len(df)} live hourly records par trained) | Error Margin (MAE): ±{mae:.2f} AQI")

c1, c2, c3 = st.columns(3)
temp_in = c1.slider("Temperature (°C)", 5.0, 48.0, float(latest['temperature_2m']))
hum_in = c2.slider("Humidity (%)", 10.0, 100.0, float(latest['relative_humidity_2m']))
wind_in = c3.slider("Wind Speed (km/h)", 0.0, 50.0, float(latest['wind_speed_10m']))

input_df = pd.DataFrame([[temp_in, hum_in, wind_in, latest['pm10'], latest['pm2_5']]], columns=features)
pred_aqi = int(model.predict(input_df)[0])

if pred_aqi <= 50:
    st.success(f"**Predicted US AQI:** {pred_aqi} — 🟢 **Good (Saaf Hawa, bahar ghoomne ke liye badhiya)**")
elif pred_aqi <= 100:
    st.warning(f"**Predicted US AQI:** {pred_aqi} — 🟡 **Moderate (Theek-Thaak, sensitive logon ko dhyan rakhna chahiye)**")
elif pred_aqi <= 150:
    st.warning(f"**Predicted US AQI:** {pred_aqi} — 🟠 **Unhealthy for Sensitive Groups**")
else:
    st.error(f"**Predicted US AQI:** {pred_aqi} — 🔴 **Poor / Unhealthy (Mask ka istemal zaroor karein!)**")

st.divider()

st.subheader(f"📈 {selected_city} Ka Pichhle 7 Din Ka Live AQI Trend")
chart_data = df.set_index("time")[["us_aqi", "pm2_5"]]
st.line_chart(chart_data)
# DataFrame ko CSV text mein badalna
csv_data = df.to_csv(index=False).encode("utf-8")

# Screen par Download Button dikhana
st.download_button(
    label=f"📥 {selected_city} Ka Live Data (CSV) Download Karein",
    data=csv_data,
    file_name=f"{selected_city.lower()}_weather_aqi_data.csv",
    mime="text/csv"
)
