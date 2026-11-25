import requests
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import export_text

# Indian Cities Coordinates (Latitude, Longitude)
CITIES = {
    "Jaipur": (26.9124, 75.7873),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),
    "Kolkata": (22.5726, 88.3639)
}

def get_live_data(city_name):
    lat, lon = CITIES[city_name]

    weather_url = "https://api.open-meteo.com/v1/forecast"
    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "past_days": 7,
        "forecast_days": 1,
        "timezone": "auto",
    }
    aqi_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm10,pm2_5,us_aqi",
        "past_days": 7,
        "forecast_days": 1,
        "timezone": "auto",
    }

    for attempt in range(3):
        try:
            w_response = requests.get(weather_url, params=weather_params, timeout=30)
            w_response.raise_for_status()
            w_res = w_response.json()

            a_response = requests.get(aqi_url, params=aqi_params, timeout=30)
            a_response.raise_for_status()
            a_res = a_response.json()

            if "hourly" not in w_res or "hourly" not in a_res:
                raise ValueError(f"Open-Meteo response is missing hourly data: {w_res} {a_res}")

            df_weather = pd.DataFrame(w_res["hourly"])
            df_aqi = pd.DataFrame(a_res["hourly"])
            df = pd.merge(df_weather, df_aqi, on="time").dropna()

            if df.empty:
                raise ValueError(f"No matching weather and AQI records returned for {city_name}")

            return df
        except (requests.exceptions.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2)

def train_aqi_model(df):
    features = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "pm10", "pm2_5"]
    X = df[features]
    y = df["us_aqi"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    mae = mean_absolute_error(y_test, model.predict(X_test))
    return model, mae, features
