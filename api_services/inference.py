from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
import joblib
import pandas as pd
import requests


# ============================================================
# 1. ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Project root:
# api_services/inference.py
#        ↑
# BASE_DIR = api_services
# PROJECT_ROOT = project root
PROJECT_ROOT = BASE_DIR.parent

# Load .env locally if it exists.
# On Vercel, Environment Variables are used automatically.
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(
    dotenv_path=ENV_FILE,
    override=False,
)


# ============================================================
# 2. MODEL PATH
# ============================================================
#
# The original project used:
#     power_anomaly_model.pkl
#
# We search several possible locations because the model may be
# located either in the project root or inside api_services.
#
# This is especially useful when deploying through GitHub/Vercel.
# ============================================================

MODEL_FILENAME = "power_anomaly_model.pkl"

MODEL_CANDIDATES = [
    PROJECT_ROOT / MODEL_FILENAME,
    BASE_DIR / MODEL_FILENAME,
    PROJECT_ROOT / "models" / MODEL_FILENAME,
    PROJECT_ROOT / "data" / MODEL_FILENAME,
]


def find_model_path() -> Path:
    """
    Find the trained Random Forest model.

    Search order:
        1. Project root
        2. api_services/
        3. models/
        4. data/

    Raises IntegrationError if the model cannot be found.
    """

    for candidate in MODEL_CANDIDATES:

        if candidate.exists() and candidate.is_file():
            return candidate

    searched_paths = "\n".join(
        f" - {path}"
        for path in MODEL_CANDIDATES
    )

    raise IntegrationError(
        "Model file not found.\n"
        f"Expected filename: {MODEL_FILENAME}\n"
        "Searched locations:\n"
        f"{searched_paths}"
    )


# ============================================================
# 3. EXTERNAL APIS
# ============================================================

OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

# ============================================================
# 4. MODEL FEATURE ORDER
# ============================================================

FEATURE_ORDER = [
    "Electricity_Consumed",
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "Avg_Past_Consumption",
    "Difference",
    "Consumption_Ratio",
    "Consumption_Change_Percentage",
    "Temp_Consumption_Interaction",
    "Heatwave_Anomaly_Risk",
]


# ============================================================
# 5. US STATE COORDINATES
# ============================================================

STATE_COORDINATES = {
    "Alabama": (33.5186, -86.8104),
    "Alaska": (61.2181, -149.9003),
    "Arizona": (33.4484, -112.0740),
    "Arkansas": (34.7465, -92.2896),
    "California": (34.0522, -118.2437),
    "Colorado": (39.7392, -104.9903),
    "Connecticut": (41.7658, -72.6734),
    "Delaware": (39.1582, -75.5244),
    "Florida": (25.7617, -80.1918),
    "Georgia": (33.7490, -84.3880),
    "Hawaii": (21.3069, -157.8583),
    "Idaho": (43.6150, -116.2023),
    "Illinois": (41.8781, -87.6298),
    "Indiana": (39.7684, -86.1581),
    "Iowa": (41.5868, -93.6250),
    "Kansas": (37.6872, -97.3301),
    "Kentucky": (38.2527, -85.7585),
    "Louisiana": (29.9511, -90.0715),
    "Maine": (43.6591, -70.2568),
    "Maryland": (39.2904, -76.6122),
    "Massachusetts": (42.3601, -71.0589),
    "Michigan": (42.3314, -83.0458),
    "Minnesota": (44.9778, -93.2650),
    "Mississippi": (32.2988, -90.1848),
    "Missouri": (38.6270, -90.1994),
    "Montana": (45.7833, -108.5007),
    "Nebraska": (41.2565, -95.9345),
    "Nevada": (36.1699, -115.1398),
    "New Hampshire": (42.9956, -71.4548),
    "New Jersey": (40.7357, -74.1724),
    "New Mexico": (35.0844, -106.6504),
    "New York": (40.7128, -74.0060),
    "North Carolina": (35.2271, -80.8431),
    "North Dakota": (46.8083, -100.7837),
    "Ohio": (39.9612, -82.9988),
    "Oklahoma": (35.4676, -97.5164),
    "Oregon": (45.5152, -122.6784),
    "Pennsylvania": (39.9526, -75.1652),
    "Rhode Island": (41.8240, -71.4128),
    "South Carolina": (34.0007, -81.0348),
    "South Dakota": (43.5460, -96.7313),
    "Tennessee": (36.1627, -86.7816),
    "Texas": (29.7604, -95.3698),
    "Utah": (40.7608, -111.8910),
    "Vermont": (44.4758, -73.2121),
    "Virginia": (36.8508, -76.2859),
    "Washington": (47.6062, -122.3321),
    "West Virginia": (38.3498, -81.6326),
    "Wisconsin": (43.0389, -87.9065),
    "Wyoming": (41.1400, -104.8202),
}


# ============================================================
# 6. CUSTOM ERROR
# ============================================================

class IntegrationError(Exception):
    pass


# ============================================================
# 7. MODEL LOADING
# ============================================================

_MODEL_CACHE = None


def load_model():
    """
    Load the trained Random Forest model.

    The model is cached after the first load so that repeated
    requests do not reload the .pkl file every time.
    """

    global _MODEL_CACHE

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    model_path = find_model_path()

    print(
        f"[PowerPulse] Loading model from: {model_path}"
    )

    try:
        _MODEL_CACHE = joblib.load(model_path)

    except Exception as error:

        raise IntegrationError(
            f"Failed to load model "
            f"'{model_path}': {error}"
        ) from error

    print(
        "[PowerPulse] Model loaded successfully."
    )

    return _MODEL_CACHE


# ============================================================
# 8. STATE COORDINATES
# ============================================================

def get_state_coordinates(
    state: str,
) -> tuple[float, float]:

    state = str(state).strip()

    if state not in STATE_COORDINATES:

        raise IntegrationError(
            f"Invalid US state: {state}"
        )

    return STATE_COORDINATES[state]


# ============================================================
# 11. OPEN-METEO WEATHER
# ============================================================

def calculate_heat_index(
    temperature_c: float,
    humidity: float,
) -> float:
    """
    Calculate heat index from temperature and relative humidity.
    Rothfusz regression, with temperature/humidity thresholds in °F/%.
    """
    temp_f = (float(temperature_c) * 9.0 / 5.0) + 32.0
    rh = float(humidity)

    # Outside the usual heat-index range, report the actual temperature.
    if temp_f < 80.0 or rh < 40.0:
        return round(float(temperature_c), 2)

    hi_f = (
        -42.379
        + 2.04901523 * temp_f
        + 10.14333127 * rh
        - 0.22475541 * temp_f * rh
        - 0.00683783 * temp_f * temp_f
        - 0.05481717 * rh * rh
        + 0.00122874 * temp_f * temp_f * rh
        + 0.00085282 * temp_f * rh * rh
        - 0.00000199 * temp_f * temp_f * rh * rh
    )

    # NOAA low-RH adjustment.
    if rh < 13.0 and 80.0 <= temp_f <= 112.0:
        adjustment = (
            ((13.0 - rh) / 4.0)
            * ((17.0 - abs(temp_f - 95.0)) / 17.0) ** 0.5
        )
        hi_f -= adjustment

    # NOAA high-RH adjustment.
    elif rh > 85.0 and 80.0 <= temp_f <= 87.0:
        adjustment = (
            ((rh - 85.0) / 10.0)
            * ((87.0 - temp_f) / 5.0)
        )
        hi_f += adjustment

    return round((hi_f - 32.0) * 5.0 / 9.0, 2)


def calculate_wet_bulb_temperature(
    temperature_c: float,
    humidity: float,
) -> float:
    """
    Stull (2011) approximation for wet-bulb temperature.
    Temperature is in °C and relative humidity in %.
    """
    temp_c = float(temperature_c)
    rh = max(0.0, min(100.0, float(humidity)))

    wet_bulb = (
        temp_c * __import__("math").atan(
            0.151977 * __import__("math").sqrt(rh + 8.313659)
        )
        + __import__("math").atan(temp_c + rh)
        - __import__("math").atan(rh - 1.676331)
        + 0.00391838
        * rh ** 1.5
        * __import__("math").atan(0.023101 * rh)
        - 4.686035
    )

    return round(float(wet_bulb), 2)


def get_current_weather(
    latitude: float,
    longitude: float,
) -> Dict[str, Any]:

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m,"
            "apparent_temperature"
        ),
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "timezone": "auto",
    }

    print(
        "[Open-Meteo] Requesting current weather..."
    )

    try:

        response = requests.get(
            OPEN_METEO_URL,
            params=params,
            timeout=30,
        )

    except requests.RequestException as error:

        raise IntegrationError(
            f"Weather API connection error: {error}"
        ) from error

    if response.status_code == 429:

        raise IntegrationError(
            "Weather API rate limit exceeded."
        )

    if response.status_code >= 400:

        raise IntegrationError(
            "Weather API error "
            f"{response.status_code}: "
            f"{response.text}"
        )

    try:

        data = response.json()

    except ValueError as error:

        raise IntegrationError(
            "Weather API returned invalid JSON."
        ) from error

    current = data.get("current")

    if not isinstance(current, dict):

        raise IntegrationError(
            "Weather API returned no current data."
        )

    temperature = current.get(
        "temperature_2m"
    )

    humidity = current.get(
        "relative_humidity_2m"
    )

    wind_speed = current.get(
        "wind_speed_10m"
    )

    apparent_temperature = current.get(
        "apparent_temperature"
    )

    timestamp = current.get(
        "time"
    )

    if any(
        value is None
        for value in (
            temperature,
            humidity,
            wind_speed,
            apparent_temperature,
            timestamp,
        )
    ):

        raise IntegrationError(
            "Incomplete weather data returned."
        )

    weather = {
        "temperature": float(
            temperature
        ),
        "humidity": float(
            humidity
        ),
        "wind_speed": float(
            wind_speed
        ),
        "apparent_temperature": float(
            apparent_temperature
        ),
        "heat_index": calculate_heat_index(
            temperature,
            humidity,
        ),
        "wet_bulb_temperature": calculate_wet_bulb_temperature(
            temperature,
            humidity,
        ),
        "timestamp": str(
            timestamp
        ),
    }

    print(
        "[Open-Meteo] Weather received: "
        f"temperature={weather['temperature']}°C, "
        f"humidity={weather['humidity']}%, "
        f"wind={weather['wind_speed']} km/h"
    )

    return weather


# ============================================================
# 12. MODEL FEATURE ENGINEERING
# ============================================================

def create_model_input(
    current_consumption: float,
    avg_past_consumption: float,
    temperature: float,
    humidity: float,
    wind_speed: float,
) -> pd.DataFrame:

    current = float(
        current_consumption
    )

    baseline = float(
        avg_past_consumption
    )

    temp = float(
        temperature
    )

    hum = float(
        humidity
    )

    wind = float(
        wind_speed
    )

    epsilon = 1e-6

    difference = (
        current - baseline
    )

    consumption_ratio = (
        current /
        (baseline + epsilon)
    )

    consumption_change_percentage = (
        difference /
        (baseline + epsilon)
    ) * 100.0

    temperature_consumption_interaction = (
        temp * current
    )

    heatwave_anomaly_risk = int(
        temp > 35.0
        and consumption_ratio > 1.10
    )

    features = {

        "Electricity_Consumed": [
            current
        ],

        "Temperature": [
            temp
        ],

        "Humidity": [
            hum
        ],

        "Wind_Speed": [
            wind
        ],

        "Avg_Past_Consumption": [
            baseline
        ],

        "Difference": [
            difference
        ],

        "Consumption_Ratio": [
            consumption_ratio
        ],

        "Consumption_Change_Percentage": [
            consumption_change_percentage
        ],

        "Temp_Consumption_Interaction": [
            temperature_consumption_interaction
        ],

        "Heatwave_Anomaly_Risk": [
            heatwave_anomaly_risk
        ],
    }

    return pd.DataFrame(
        features,
        columns=FEATURE_ORDER,
    )


# ============================================================
# 14. MODEL PREDICTION
# ============================================================

def predict_anomaly(
    model,
    model_input: pd.DataFrame,
    current_consumption: float,
    avg_past_consumption: float,
) -> Dict[str, Any]:

    current = float(
        current_consumption
    )

    baseline = float(
        avg_past_consumption
    )

    epsilon = 1e-6

    ratio = (
        current /
        (baseline + epsilon)
    )

    # ========================================================
    # NORMAL RANGE
    # ========================================================

    if 0.80 <= ratio <= 1.20:

        return {

            "prediction": 0,

            "prediction_status": "Normal",

            "label": "Normal",

            "risk_level": "LOW",

            "risk_score": 0.0,

            "action": (
                "Maintain routine operational monitoring"
            ),

            "action_code": "MONITOR",

            "abnormal_probability": 0.0,

            "consumption_ratio": round(
                ratio,
                4,
            ),
        }

    # ========================================================
    # CRITICAL UNDER-CONSUMPTION
    # ========================================================

    if (
        current <= 0.0
        or ratio < 0.25
    ):

        return {

            "prediction": 1,

            "prediction_status": "Abnormal",

            "label": "Abnormal",

            "risk_level": "CRITICAL",

            "risk_score": 100.0,

            "action": (
                "Mandatory deployment of emergency "
                "field crews required due to high "
                "probability of illicit bypass connections."
            ),

            "action_code": "CRITICAL_INSPECT",

            "abnormal_probability": 1.0,

            "consumption_ratio": round(
                ratio,
                4,
            ),
        }

    # ========================================================
    # RANDOM FOREST PREDICTION
    # ========================================================

    try:

        model_prediction = int(
            model.predict(
                model_input
            )[0]
        )

    except Exception as error:

        raise IntegrationError(
            f"Model prediction failed: {error}"
        ) from error

    # ========================================================
    # PROBABILITY
    # ========================================================

    if hasattr(
        model,
        "predict_proba",
    ):

        try:

            probabilities = (
                model.predict_proba(
                    model_input
                )[0]
            )

            classes = list(
                getattr(
                    model,
                    "classes_",
                    [0, 1],
                )
            )

            if 1 in classes:

                abnormal_index = (
                    classes.index(1)
                )

                probability = float(
                    probabilities[
                        abnormal_index
                    ]
                )

            else:

                probability = 0.0

        except Exception as error:

            raise IntegrationError(
                "Model probability prediction failed: "
                f"{error}"
            ) from error

    else:

        probability = (
            1.0
            if model_prediction == 1
            else 0.0
        )

    probability = round(
        probability,
        4,
    )

    # ========================================================
    # FINAL PREDICTION
    # ========================================================

    if probability <= 0.50:

        final_prediction = 0

        prediction_status = "Normal"

        label = "Normal"

    else:

        final_prediction = 1

        prediction_status = (
            "Potentially Abnormal"
        )

        label = "Abnormal"

    # ========================================================
    # RISK
    # ========================================================

    if (
        1.20 < ratio <= 2.00
        or 0.50 <= ratio < 0.80
    ):

        risk_level = "MEDIUM"

        risk_score = 50.0

        action_code = "INVESTIGATE"

        action = (
            "Conduct remote telemetry audits "
            "of smart meter logs in correlation "
            "with localized meteorological data."
        )

    elif (
        2.00 < ratio <= 4.00
        or 0.25 <= ratio < 0.50
    ):

        risk_level = "HIGH"

        risk_score = 75.0

        action_code = "INSPECT"

        action = (
            "Mandatory dispatch of a field audit "
            "team to conduct a physical meter "
            "examination and diagnostic assessment."
        )

    else:

        risk_level = "CRITICAL"

        risk_score = 100.0

        action_code = "CRITICAL_INSPECT"

        action = (
            "Urgent mobilization of technical "
            "emergency personnel alongside legal "
            "liability procedures."
        )

    # ========================================================
    # NORMAL PREDICTION OVERRIDE
    # ========================================================

    if final_prediction == 0:

        prediction_status = "Normal"

        label = "Normal"

        risk_level = "LOW"

        risk_score = 0.0

        action_code = "MONITOR"

        action = (
            "Maintain routine operational monitoring"
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

        "consumption_ratio": round(
            ratio,
            4,
        ),
    }


# ============================================================
# 15. COMPLETE POWERGUARD ANALYSIS
# ============================================================

def run_powerguard_analysis(
    state: str,
    current_consumption: float,
    avg_past_consumption: float,
) -> Dict[str, Any]:

    current_consumption = float(
        current_consumption
    )

    avg_past_consumption = float(
        avg_past_consumption
    )

    if current_consumption < 0:

        raise IntegrationError(
            "Current consumption cannot be negative."
        )

    if avg_past_consumption <= 0:

        raise IntegrationError(
            "Historical baseline consumption "
            "must be greater than zero."
        )

    # ========================================================
    # 1. LOCATION
    # ========================================================

    latitude, longitude = (
        get_state_coordinates(
            state
        )
    )

    # ========================================================
    # 2. WEATHER
    # ========================================================

    weather = get_current_weather(
        latitude,
        longitude,
    )

    # ========================================================
    # 3. MODEL INPUT
    # ========================================================

    model_input = create_model_input(
        current_consumption=current_consumption,
        avg_past_consumption=avg_past_consumption,
        temperature=float(weather["temperature"]),
        humidity=float(weather["humidity"]),
        wind_speed=weather["wind_speed"],
    )

    # ========================================================
    # 6. LOAD MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # 7. PREDICTION
    # ========================================================

    prediction = predict_anomaly(
        model=model,
        model_input=model_input,
        current_consumption=current_consumption,
        avg_past_consumption=avg_past_consumption,
    )

    print(
        "[PowerPulse] Model prediction: "
        f"{prediction['label']} "
        f"probability="
        f"{prediction['abnormal_probability']}"
    )

    # ========================================================
    # 8. FINAL RESPONSE
    # ========================================================

    return {

        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------

        "location": {

            "state": state,

            "latitude": latitude,

            "longitude": longitude,
        },

        # ----------------------------------------------------
        # WEATHER
        # ----------------------------------------------------

        "weather": {

            "temperature_c": float(
                weather["temperature"]
            ),

            "humidity_percent": float(
                weather["humidity"]
            ),

            "wind_speed_kmh": float(
                weather["wind_speed"]
            ),

            "heat_index_c": float(
                weather["heat_index"]
            ),

            "apparent_temperature_c": float(
                weather["apparent_temperature"]
            ),

            "wet_bulb_temperature_c": float(
                weather["wet_bulb_temperature"]
            ),

            "timestamp": weather[
                "timestamp"
            ],
        },

        # ----------------------------------------------------
        # ENVIRONMENTAL INTELLIGENCE
        # ----------------------------------------------------

        "environmental_intelligence": {
            "heat_index_c": float(weather["heat_index"]),
            "apparent_temperature_c": float(
                weather["apparent_temperature"]
            ),
            "wet_bulb_temperature_c": float(
                weather["wet_bulb_temperature"]
            ),
        },

        # ----------------------------------------------------
        # CONSUMPTION
        # ----------------------------------------------------

        "consumption": {

            "current_kwh": (
                current_consumption
            ),

            "historical_baseline_kwh": (
                avg_past_consumption
            ),
        },

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        "model": {

            "prediction": (
                prediction[
                    "prediction"
                ]
            ),

            "prediction_status": (
                prediction[
                    "prediction_status"
                ]
            ),

            "label": (
                prediction[
                    "label"
                ]
            ),

            "risk_level": (
                prediction[
                    "risk_level"
                ]
            ),

            "risk_score": (
                prediction[
                    "risk_score"
                ]
            ),

            "action": (
                prediction[
                    "action"
                ]
            ),

            "action_code": (
                prediction[
                    "action_code"
                ]
            ),

            "abnormal_probability": (
                prediction[
                    "abnormal_probability"
                ]
            ),

            "consumption_ratio": (
                prediction[
                    "consumption_ratio"
                ]
            ),

            "features": (
                model_input.to_dict(
                    orient="records"
                )[0]
            ),
        },
    }
