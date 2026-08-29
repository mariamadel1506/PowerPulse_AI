import os
import time
import pickle
import datetime
import numpy as np
import pandas as pd
import requests
from typing import Dict, Any, Optional

# ==========================================
# CUSTOM EXCEPTIONS
# ==========================================
class IntegrationError(Exception):
    """Custom exception raised during integration processing failures."""
    pass


# ==========================================
# GLOBAL CONSTANTS & API ENDPOINTS
# ==========================================
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
FORTYGUARD_ENV_URL = "https://api.fortyguard.com/v1/env_params"
FORTYGUARD_STATUS_URL = "https://api.fortyguard.com/v1/status"

EXPECTED_FEATURES = [
    'Temperature',
    'Humidity',
    'Pressure',
    'Wind_Speed',
    'Industrial_Activity_Index',
    'Solar_Radiation',
    'Previous_Hour_Consumption',
    'Consumption_Ratio',
    'Heatwave_Anomaly_Risk',
    'Historical_Consumption_Std',
    'Voltage_Fluctuation_Index',
    'Grid_Stress_Factor',
    'Peak_Hour_Flag'
]


# ==========================================
# MODEL LOADER
# ==========================================
def load_model(model_path: str = "models/xgboost_model.pkl"):
    """Loads the pre-trained XGBoost model from disk."""
    if not os.path.exists(model_path):
        raise IntegrationError(f"Model file not found at path: {model_path}")

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as error:
        raise IntegrationError(f"Failed to load XGBoost model: {error}") from error


GLOBAL_MODEL = None
try:
    GLOBAL_MODEL = load_model()
except Exception:
    pass


# ==========================================
# EXTERNAL API INTEGRATIONS
# ==========================================
def fetch_weather_data(latitude: float, longitude: float) -> Dict[str, float]:
    """Fetches real-time weather metrics from Open-Meteo API."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "direct_normal_irradiance"
        ],
        "timezone": "auto"
    }

    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        current = data.get("current", {})
        return {
            "temperature": float(current.get("temperature_2m", 25.0)),
            "humidity": float(current.get("relative_humidity_2m", 50.0)),
            "pressure": float(current.get("surface_pressure", 1013.25)),
            "wind_speed": float(current.get("wind_speed_10m", 10.0)),
            "solar_radiation": float(current.get("direct_normal_irradiance", 0.0))
        }
    except Exception as error:
        raise IntegrationError(f"Failed to fetch weather data: {error}") from error


def fetch_fortyguard_temperature(
    latitude: float,
    longitude: float,
    api_key: Optional[str] = None,
    current_temp: Optional[float] = None
) -> float:
    """
    Submits environmental parameters job to FortyGuard API and polls for completed results.
    Falls back to Open-Meteo temperature if request or key is unavailable.
    """
    if not api_key:
        api_key = os.getenv("FORTYGUARD_API_KEY")

    if not api_key:
        if current_temp is not None:
            return float(current_temp)
        weather = fetch_weather_data(latitude, longitude)
        return weather["temperature"]

    headers = {
        "api-key": api_key,
        "Content-Type": "application/json"
    }

    now = datetime.datetime.now(datetime.timezone.utc)
    today_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    base_temp = current_temp if current_temp is not None else 25.0

    payload = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "temperature": float(base_temp),
        "date_time": {
            "start_date": today_str,
            "start_time": time_str,
            "filter_type": 1
        }
    }

    try:
        # 1. إرسال طلب التحليل
        res = requests.post(FORTYGUARD_ENV_URL, headers=headers, json=payload, timeout=10)
        if res.status_code != 200:
            return base_temp
        
        init_data = res.json()
        activity_id = init_data.get("data", {}).get("activity_id")
        if not activity_id:
            return base_temp

        # 2. Polling للحصول على النتيجة المنتظرة
        status_url = f"{FORTYGUARD_STATUS_URL}/{activity_id}"
        max_attempts = 5
        for _ in range(max_attempts):
            time.sleep(1)
            poll_res = requests.get(status_url, headers=headers, timeout=10)
            if poll_res.status_code == 200:
                pdata = poll_res.json()
                if pdata.get("message") == "Completed" or pdata.get("data", {}).get("status") == "Completed":
                    result = pdata.get("data", {}).get("result", {})
                    locations = result.get("locations", [])
                    if locations and "temperature" in locations[0]:
                        return float(locations[0]["temperature"])
                    break

        return base_temp

    except Exception:
        return base_temp


# ==========================================
# FEATURE PREPARATION
# ==========================================
def prepare_model_input(
    weather_data: Dict[str, float],
    fortyguard_temp: float,
    current_consumption: float,
    avg_past_consumption: float,
    prev_hour_consumption: Optional[float] = None,
    industrial_activity_index: float = 0.5,
    historical_consumption_std: float = 10.0,
    voltage_fluctuation_index: float = 0.02,
    grid_stress_factor: float = 0.5,
    peak_hour_flag: int = 0
) -> pd.DataFrame:
    """
    Transforms raw environmental, system, and historical inputs
    into a structured DataFrame ready for XGBoost inference.
    """
    try:
        current = float(current_consumption)
        baseline = float(avg_past_consumption)
        epsilon = 1e-6

        prev_hour = (
            float(prev_hour_consumption)
            if prev_hour_consumption is not None
            else current
        )

        effective_temp = float(fortyguard_temp)
        ratio = current / (baseline + epsilon)

        if effective_temp > 35.0 and ratio > 1.2:
            heatwave_risk = min(1.0, (effective_temp - 35.0) * 0.1 * ratio)
        else:
            heatwave_risk = 0.0

        features_dict = {
            'Temperature': effective_temp,
            'Humidity': float(weather_data.get('humidity', 50.0)),
            'Pressure': float(weather_data.get('pressure', 1013.25)),
            'Wind_Speed': float(weather_data.get('wind_speed', 10.0)),
            'Industrial_Activity_Index': float(industrial_activity_index),
            'Solar_Radiation': float(weather_data.get('solar_radiation', 0.0)),
            'Previous_Hour_Consumption': prev_hour,
            'Consumption_Ratio': round(ratio, 4),
            'Heatwave_Anomaly_Risk': round(heatwave_risk, 4),
            'Historical_Consumption_Std': float(historical_consumption_std),
            'Voltage_Fluctuation_Index': float(voltage_fluctuation_index),
            'Grid_Stress_Factor': float(grid_stress_factor),
            'Peak_Hour_Flag': int(peak_hour_flag)
        }

        df_input = pd.DataFrame([features_dict])
        df_input = df_input[EXPECTED_FEATURES]
        return df_input

    except Exception as error:
        raise IntegrationError(f"Failed to prepare model input: {error}") from error


# ==========================================
# ANOMALY PREDICTION ENGINE
# ==========================================
def predict_anomaly(
    model,
    model_input: pd.DataFrame,
    current_consumption: float,
    avg_past_consumption: float
) -> Dict[str, Any]:
    """
    Runs XGBoost inference, evaluates consumption deviation,
    and returns a structured payload for the UI/API response.
    """
    current = float(current_consumption)
    baseline = float(avg_past_consumption)
    epsilon = 1e-6
    ratio = current / (baseline + epsilon)

    if current <= 0.0 or ratio < 0.25:
        return {
            "prediction": 1,
            "prediction_status": "Abnormal",
            "label": "Abnormal",
            "risk_level": "CRITICAL",
            "risk_score": 100.0,
            "action": (
                "Mandatory deployment of emergency field crews required "
                "due to high probability of illicit bypass connections."
            ),
            "action_code": "CRITICAL_INSPECT",
            "abnormal_probability": 1.0,
            "consumption_ratio": round(ratio, 4)
        }

    try:
        model_prediction = int(model.predict(model_input)[0])
    except Exception as error:
        raise IntegrationError(f"Model prediction failed: {error}") from error

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(model_input)[0]
            classes = list(getattr(model, "classes_", [0, 1]))
            abnormal_index = classes.index(1) if 1 in classes else 0
            probability = float(probabilities[abnormal_index])
        except Exception as error:
            raise IntegrationError(f"Model probability prediction failed: {error}") from error
    else:
        probability = 1.0 if model_prediction == 1 else 0.0

    probability = round(probability, 4)

    final_prediction = 1 if probability > 0.50 else 0

    if final_prediction == 0:
        prediction_status = "Normal"
        label = "Normal"
        risk_level = "LOW"
        risk_score = 0.0
        action_code = "MONITOR"
        action = "Maintain routine operational monitoring"
    else:
        prediction_status = "Potentially Abnormal"
        label = "Abnormal"

        if 1.20 < ratio <= 2.00 or 0.50 <= ratio < 0.80:
            risk_level = "MEDIUM"
            risk_score = 50.0
            action_code = "INVESTIGATE"
            action = (
                "Conduct remote telemetry audits of smart meter logs "
                "in correlation with localized meteorological data."
            )
        elif 2.00 < ratio <= 4.00 or 0.25 <= ratio < 0.50:
            risk_level = "HIGH"
            risk_score = 75.0
            action_code = "INSPECT"
            action = (
                "Mandatory dispatch of a field audit team to conduct "
                "a physical meter examination and diagnostic assessment."
            )
        else:
            risk_level = "CRITICAL"
            risk_score = 100.0
            action_code = "CRITICAL_INSPECT"
            action = (
                "Urgent mobilization of technical emergency personnel "
                "alongside legal liability procedures."
            )

    return {
        "prediction": final_prediction,
        "prediction_status": prediction_status,
        "label": label,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "action": action,
        "action_code": action_code,
        "abnormal_probability": probability,
        "consumption_ratio": round(ratio, 4)
    }


# ==========================================
# MAIN PIPELINE EXECUTION
# ==========================================
def run_powerguard_analysis(
    latitude: float = 0.0,
    longitude: float = 0.0,
    current_consumption: float = 0.0,
    avg_past_consumption: float = 0.0,
    prev_hour_consumption: Optional[float] = None,
    industrial_activity_index: float = 0.5,
    historical_consumption_std: float = 10.0,
    voltage_fluctuation_index: float = 0.02,
    grid_stress_factor: float = 0.5,
    peak_hour_flag: int = 0,
    fortyguard_api_key: Optional[str] = None,
    model_path: str = "models/xgboost_model.pkl",
    state: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main orchestration function. Accepts 'state' and **kwargs dynamically
    to prevent UI parameter incompatibility errors.
    """
    model = GLOBAL_MODEL if GLOBAL_MODEL is not None else load_model(model_path)

    weather_data = fetch_weather_data(latitude, longitude)
    fortyguard_temp = fetch_fortyguard_temperature(
        latitude,
        longitude,
        fortyguard_api_key,
        current_temp=weather_data.get("temperature")
    )

    model_input = prepare_model_input(
        weather_data=weather_data,
        fortyguard_temp=fortyguard_temp,
        current_consumption=current_consumption,
        avg_past_consumption=avg_past_consumption,
        prev_hour_consumption=prev_hour_consumption,
        industrial_activity_index=industrial_activity_index,
        historical_consumption_std=historical_consumption_std,
        voltage_fluctuation_index=voltage_fluctuation_index,
        grid_stress_factor=grid_stress_factor,
        peak_hour_flag=peak_hour_flag
    )

    analysis_result = predict_anomaly(
        model=model,
        model_input=model_input,
        current_consumption=current_consumption,
        avg_past_consumption=avg_past_consumption
    )

    full_payload = {
        "status": "success",
        "location": {
            "latitude": latitude,
            "longitude": longitude,
            "state": state
        },
        "environmental_metrics": {
            "open_meteo_temp": weather_data.get("temperature"),
            "fortyguard_temp": fortyguard_temp,
            "humidity": weather_data.get("humidity"),
            "pressure": weather_data.get("pressure"),
            "wind_speed": weather_data.get("wind_speed"),
            "solar_radiation": weather_data.get("solar_radiation")
        },
        "consumption_metrics": {
            "current_consumption": float(current_consumption),
            "avg_past_consumption": float(avg_past_consumption),
            "consumption_ratio": analysis_result["consumption_ratio"]
        },
        "analysis": analysis_result
    }

    return full_payload
