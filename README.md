# 🌤️ Real-Time Indian Cities Weather & AQI Predictor (AI/ML)

An end-to-end real-time Machine Learning web application that fetches live weather and air quality data for major Indian cities (Jaipur, Delhi, Mumbai, Bangalore, Kolkata) and predicts the US Air Quality Index (AQI) using a Random Forest Regressor.

## 🚀 Features
- **Live Data Ingestion:** Fetches real-time hourly weather (`temperature`, `humidity`, `wind speed`) and pollution (`PM10`, `PM2.5`, `US AQI`) metrics via the Open-Meteo API (no API key required).
- **Automated ML Pipeline:** Cleans and merges live datasets using `Pandas` and trains a `RandomForestRegressor` model dynamically.
- **Interactive What-If Simulator:** Adjust temperature, humidity, and wind speed sliders to simulate real-time AQI changes with automated health advisories.
- **Historical Trend Visualization:** Interactive 7-day line chart tracking `US AQI` vs `PM2.5`.

## 🛠️ Tech Stack
- **Language:** Python
- **Data Processing:** Pandas, Requests
- **Machine Learning:** Scikit-Learn (Random Forest Regressor)
- **Frontend / Deployment:** Streamlit

## 💻 How to Run Locally
```bash
git clone https://github.com/<your-username>/live-weather-aqi-predictor.git
cd live-weather-aqi-predictor
pip install -r requirements.txt
streamlit run app.py
```
