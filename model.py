import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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
    
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        f"&past_days=7&forecast_days=1"
    )
    aqi_url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}&hourly=pm10,pm2_5,us_aqi"
        f"&past_days=7&forecast_days=1"
    )
    
    w_res = requests.get(weather_url, timeout=15).json()
    a_res = requests.get(aqi_url, timeout=15).json()
    
    df_weather = pd.DataFrame(w_res["hourly"])
    df_aqi = pd.DataFrame(a_res["hourly"])
    
    df = pd.merge(df_weather, df_aqi, on="time").dropna()
    return df

def train_aqi_model(df):
    features = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "pm10", "pm2_5"]
    X = df[features]
    y = df["us_aqi"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    mae = mean_absolute_error(y_test, model.predict(X_test))
    return model, mae, features
