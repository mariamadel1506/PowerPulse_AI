from datetime import datetime
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

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

FORTYGUARD_URL = os.getenv(
    "FORTYGUARD_BASE_URL",
    "https://api.fortyguard.com/v1",
).strip().rstrip("/")


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
# 8. FORTYGUARD API KEY
# ============================================================

def get_api_key() -> str:
    """
    Read FortyGuard API key.

    Locally:
        .env

    Vercel:
        Environment Variables
    """

    api_key = os.getenv(
        "FORTYGUARD_API_KEY",
        "",
    ).strip()

    if not api_key:

        raise IntegrationError(
            "FORTYGUARD_API_KEY is missing. "
            "Add it to your local .env file or "
            "Vercel Environment Variables."
        )

    return api_key


# ============================================================
# 9. FORTYGUARD HEADERS
# ============================================================

def get_fortyguard_headers() -> Dict[str, str]:
    """
    FortyGuard documentation specifies `api-key`.
    """

    return {
        "api-key": get_api_key(),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# ============================================================
# 10. STATE COORDINATES
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
            "wind_speed_10m"
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

    timestamp = current.get(
        "time"
    )

    if any(
        value is None
        for value in (
            temperature,
            humidity,
            wind_speed,
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
# 12. FORTYGUARD SUBMISSION
# ============================================================

def submit_fortyguard(
    latitude: float,
    longitude: float,
    temperature: float,
    timestamp: str,
) -> str:

    try:

        clean_timestamp = str(
            timestamp
        ).replace(
            "Z",
            "+00:00",
        )

        timestamp_dt = datetime.fromisoformat(
            clean_timestamp
        )

    except ValueError as error:

        raise IntegrationError(
            f"Invalid weather timestamp: {error}"
        ) from error

    payload = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "temperature": float(temperature),
        "date_time": {
            "start_date": timestamp_dt.strftime(
                "%Y-%m-%d"
            ),
            "start_time": timestamp_dt.replace(
                minute=0,
                second=0,
                microsecond=0,
            ).strftime("%H:%M"),
            "filter_type": 1,
        },
    }

    url = (
        f"{FORTYGUARD_URL}/env_params"
    )

    print(
        "[FortyGuard] Submitting environment analysis..."
    )

    print(
        f"[FortyGuard] URL: {url}"
    )

    print(
        "[FortyGuard] Payload: "
        f"latitude={payload['latitude']}, "
        f"longitude={payload['longitude']}, "
        f"temperature={payload['temperature']}, "
        f"date={payload['date_time']['start_date']}, "
        f"time={payload['date_time']['start_time']}"
    )

    try:

        response = requests.post(
            url,
            headers=get_fortyguard_headers(),
            json=payload,
            timeout=60,
        )

    except requests.RequestException as error:

        raise IntegrationError(
            f"FortyGuard connection error: {error}"
        ) from error

    try:

        response_data = response.json()

    except ValueError:

        response_data = None

    if response.status_code >= 400:

        body = (
            response_data
            if response_data is not None
            else response.text
        )

        raise IntegrationError(
            "FortyGuard submission failed "
            f"(HTTP {response.status_code}): "
            f"{body}"
        )

    if not isinstance(
        response_data,
        dict,
    ):

        raise IntegrationError(
            "FortyGuard returned an unexpected "
            f"response: {response.text}"
        )

    if response_data.get(
        "error"
    ) is True:

        raise IntegrationError(
            "FortyGuard API rejected the request: "
            f"{response_data.get('message', response_data)}"
        )

    response_data_inner = (
        response_data.get("data")
    )

    if not isinstance(
        response_data_inner,
        dict,
    ):

        response_data_inner = {}

    activity_id = (
        response_data_inner.get(
            "activity_id"
        )
        or response_data.get(
            "activity_id"
        )
    )

    if not activity_id:

        raise IntegrationError(
            "FortyGuard did not return activity_id. "
            f"Response: {response_data}"
        )

    activity_id = str(
        activity_id
    )

    print(
        "[FortyGuard] Activity submitted successfully: "
        f"{activity_id}"
    )

    return activity_id


# ============================================================
# 13. FORTYGUARD STATUS / POLLING
# ============================================================

def get_fortyguard_result(
    activity_id: str,
    max_attempts: int = 24,
    wait_seconds: int = 5,
) -> Dict[str, Any]:

    activity_id = str(
        activity_id
    ).strip()

    if not activity_id:
        raise IntegrationError(
            "FortyGuard activity_id is empty."
        )

    url = (
        f"{FORTYGUARD_URL}"
        f"/status/{activity_id}"
    )

    print(
        "[FortyGuard] Polling activity: "
        f"{activity_id}"
    )

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        try:
            response = requests.get(
                url,
                headers=get_fortyguard_headers(),
                timeout=20,
            )

        except requests.RequestException as error:
            # A temporary status-network failure should not create
            # another activity. Re-check the SAME activity.
            print(
                "[FortyGuard] Temporary status error: "
                f"{error}. Retrying same activity..."
            )

            if attempt < max_attempts:
                time.sleep(wait_seconds)
                continue

            raise IntegrationError(
                "FortyGuard status connection failed after "
                f"{max_attempts} checks. "
                f"activity_id={activity_id}"
            ) from error

        try:
            response_data = response.json()
        except ValueError as error:
            print(
                "[FortyGuard] Invalid JSON on status check; "
                "retrying same activity."
            )

            if attempt < max_attempts:
                time.sleep(wait_seconds)
                continue

            raise IntegrationError(
                "FortyGuard status returned invalid JSON. "
                f"activity_id={activity_id}"
            ) from error

        # Rate limits / transient server errors: do not submit a
        # second activity; back off and poll the same activity.
        if response.status_code in {429, 500, 502, 503, 504}:
            print(
                "[FortyGuard] Temporary HTTP "
                f"{response.status_code}; retrying same activity."
            )

            if attempt < max_attempts:
                time.sleep(wait_seconds)
                continue

            raise IntegrationError(
                "FortyGuard status remained unavailable. "
                f"HTTP {response.status_code}; "
                f"activity_id={activity_id}"
            )

        if response.status_code >= 400:
            raise IntegrationError(
                "FortyGuard status request failed "
                f"(HTTP {response.status_code}): "
                f"{response_data}"
            )

        if not isinstance(
            response_data,
            dict,
        ):
            raise IntegrationError(
                "FortyGuard status response is not an object."
            )

        if response_data.get("error") is True:
            raise IntegrationError(
                "FortyGuard status API error: "
                f"{response_data.get('message', response_data)}"
            )

        data = response_data.get("data")

        if not isinstance(data, dict):
            data = response_data

        status = str(
            data.get("status")
            or response_data.get("status")
            or ""
        ).strip().lower()

        print(
            "[FortyGuard] "
            f"check {attempt}/{max_attempts} "
            f"status={status}"
        )

        if status in {
            "completed",
            "complete",
            "success",
            "succeeded",
            "done",
            "ok",
        }:
            result = data.get("result")

            if not isinstance(result, dict):
                raise IntegrationError(
                    "FortyGuard returned Completed "
                    "but result object is missing. "
                    f"activity_id={activity_id}"
                )

            return data

        if status in {
            "failed",
            "failure",
            "error",
        }:
            error_message = (
                data.get("message")
                or data.get("error")
                or response_data.get("message")
                or "FortyGuard activity failed."
            )

            raise IntegrationError(
                "FortyGuard processing error: "
                f"{error_message}; "
                f"activity_id={activity_id}"
            )

        if attempt < max_attempts:
            time.sleep(wait_seconds)

    raise TimeoutError(
        "FortyGuard activity did not complete "
        f"within {max_attempts * wait_seconds} seconds. "
        f"activity_id={activity_id}"
    )


# ============================================================
# 14. SAFE VALUE EXTRACTION
# ============================================================

def _first_non_null_value(
    obj: Dict[str, Any],
    keys: list[str],
) -> Optional[Any]:

    def first_value(value: Any) -> Optional[Any]:

        if value is None:
            return None

        if isinstance(value, list):
            for item in value:
                found = first_value(item)
                if found is not None:
                    return found
            return None

        if isinstance(value, dict):
            # Some API responses may wrap the actual array/value.
            for nested_key in (
                "value",
                "values",
                "data",
            ):
                if nested_key in value:
                    found = first_value(
                        value.get(nested_key)
                    )
                    if found is not None:
                        return found
            return None

        return value

    for key in keys:

        if key not in obj:
            continue

        found = first_value(
            obj.get(key)
        )

        if found is not None:
            return found

    return None


def _safe_float(
    value: Any,
) -> Optional[float]:

    if value is None:
        return None

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):

        return None


# ============================================================
# 15. EXTRACT FORTYGUARD INTELLIGENCE
# ============================================================

def extract_fortyguard_intelligence(
    data: Dict[str, Any],
) -> Dict[str, Any]:

    if not isinstance(
        data,
        dict,
    ):

        raise IntegrationError(
            "FortyGuard response is not a dictionary."
        )

    result = data.get(
        "result"
    )

    if not isinstance(
        result,
        dict,
    ):

        result = data

    locations = result.get(
        "locations"
    )

    if (
        not isinstance(
            locations,
            list,
        )
        or not locations
    ):

        single_location = result.get(
            "location"
        )

        if isinstance(
            single_location,
            dict,
        ):

            locations = [
                single_location
            ]

        else:

            raise IntegrationError(
                "FortyGuard returned no valid locations."
            )

    location = locations[0]

    if not isinstance(
        location,
        dict,
    ):

        raise IntegrationError(
            "FortyGuard location structure is invalid."
        )

    parameters = location.get(
        "parameters"
    )

    if not isinstance(
        parameters,
        dict,
    ):

        parameters = {}

    # ========================================================
    # TEMPERATURE
    # ========================================================

    temperature = _first_non_null_value(
        location,
        [
            "temperature",
            "temp",
            "temperature_celsius",
        ],
    )

    if temperature is None:

        temperature = _first_non_null_value(
            parameters,
            [
                "temperature",
                "temp",
                "temperature_celsius",
            ],
        )

    # ========================================================
    # LOCATION
    # ========================================================

    latitude = _first_non_null_value(
        location,
        [
            "lat",
            "latitude",
        ],
    )

    longitude = _first_non_null_value(
        location,
        [
            "lon",
            "longitude",
        ],
    )

    # ========================================================
    # HUMIDITY
    # ========================================================

    humidity = _first_non_null_value(
        parameters,
        [
            "relative_humidity_percent",
            "relative_humidity",
            "humidity",
        ],
    )

    # ========================================================
    # HEAT INDEX
    # ========================================================

    heat_index = _first_non_null_value(
        parameters,
        [
            "heat_index_celsius",
            "heat_index",
        ],
    )

    # ========================================================
    # APPARENT TEMPERATURE
    # ========================================================

    apparent_temperature = _first_non_null_value(
        parameters,
        [
            "apparent_temperature_celsius",
            "apparent_temperature",
            "feels_like",
        ],
    )

    # ========================================================
    # WET BULB
    # ========================================================

    wet_bulb_temperature = _first_non_null_value(
        parameters,
        [
            "wet_bulb_temperature_celsius",
            "wet_bulb_temperature",
            "wet_bulb",
        ],
    )

    intelligence = {

        "temperature": _safe_float(
            temperature
        ),

        "latitude": _safe_float(
            latitude
        ),

        "longitude": _safe_float(
            longitude
        ),

        "humidity": _safe_float(
            humidity
        ),

        "heat_index": _safe_float(
            heat_index
        ),

        "apparent_temperature": _safe_float(
            apparent_temperature
        ),

        "wet_bulb_temperature": _safe_float(
            wet_bulb_temperature
        ),
    }

    print(
        "[FortyGuard] Extracted intelligence:"
    )

    print(
        f"  temperature={intelligence['temperature']}"
    )

    print(
        f"  humidity={intelligence['humidity']}"
    )

    print(
        f"  heat_index={intelligence['heat_index']}"
    )

    print(
        "  apparent_temperature="
        f"{intelligence['apparent_temperature']}"
    )

    print(
        "  wet_bulb_temperature="
        f"{intelligence['wet_bulb_temperature']}"
    )

    return intelligence


# ============================================================
# 16. MODEL FEATURE ENGINEERING
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
# 17. MODEL PREDICTION
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
# 18. COMPLETE POWERGUARD ANALYSIS
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
    # 3. FORTYGUARD
    # ========================================================

    print(
        "[PowerPulse] Starting FortyGuard integration..."
    )

    activity_id = submit_fortyguard(
        latitude=latitude,
        longitude=longitude,
        temperature=weather["temperature"],
        timestamp=weather["timestamp"],
    )

    fortyguard_data = get_fortyguard_result(
        activity_id=activity_id,
        max_attempts=30,
        wait_seconds=2,
    )

    thermal_intelligence = (
        extract_fortyguard_intelligence(
            fortyguard_data
        )
    )

    # If the first completed payload is structurally present but
    # environmental values are unexpectedly empty, re-read the
    # SAME completed activity once. This does not create another
    # FortyGuard job and prevents the repeated-activity problem.
    if all(
        thermal_intelligence.get(name) is None
        for name in (
            "humidity",
            "heat_index",
            "apparent_temperature",
            "wet_bulb_temperature",
        )
    ):
        print(
            "[PowerPulse] FortyGuard completed but returned "
            "empty environmental arrays. Re-reading the same "
            "activity once..."
        )

        time.sleep(2)

        try:
            refreshed_data = get_fortyguard_result(
                activity_id=activity_id,
                max_attempts=4,
                wait_seconds=3,
            )

            refreshed_intelligence = (
                extract_fortyguard_intelligence(
                    refreshed_data
                )
            )

            if any(
                refreshed_intelligence.get(name) is not None
                for name in (
                    "humidity",
                    "heat_index",
                    "apparent_temperature",
                    "wet_bulb_temperature",
                )
            ):
                fortyguard_data = refreshed_data
                thermal_intelligence = refreshed_intelligence

        except Exception as refresh_error:
            print(
                "[PowerPulse] Same-activity refresh did not "
                f"change the result: {refresh_error}"
            )

    print(
        "[PowerPulse] FortyGuard completed successfully."
    )

    # ========================================================
    # 4. ENVIRONMENT DATA
    # ========================================================
    #
    # IMPORTANT:
    #
    # FortyGuard can return null for individual fields.
    #
    # This does NOT mean FortyGuard failed.
    #
    # If FortyGuard has no humidity, use Open-Meteo humidity.
    # If FortyGuard has no temperature, use Open-Meteo temperature.
    #
    # FortyGuard itself remains ACTIVE in the response.
    # ========================================================

    fortyguard_temperature = (
        thermal_intelligence.get(
            "temperature"
        )
    )

    fortyguard_humidity = (
        thermal_intelligence.get(
            "humidity"
        )
    )

    # Temperature fallback
    if fortyguard_temperature is None:

        model_temperature = float(
            weather["temperature"]
        )

        print(
            "[PowerPulse] FortyGuard did not "
            "return temperature. "
            "Using Open-Meteo temperature "
            "for model input."
        )

    else:

        model_temperature = float(
            fortyguard_temperature
        )

    # Humidity fallback
    if fortyguard_humidity is None:

        model_humidity = float(
            weather["humidity"]
        )

        print(
            "[PowerPulse] FortyGuard did not "
            "return relative humidity. "
            "Using Open-Meteo humidity "
            "for model input."
        )

    else:

        model_humidity = float(
            fortyguard_humidity
        )

    # ========================================================
    # 5. MODEL INPUT
    # ========================================================

    model_input = create_model_input(
        current_consumption=current_consumption,
        avg_past_consumption=avg_past_consumption,
        temperature=model_temperature,
        humidity=model_humidity,
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

            "timestamp": weather[
                "timestamp"
            ],
        },

        # ----------------------------------------------------
        # FORTYGUARD
        # ----------------------------------------------------

        "fortyguard": {

            # FortyGuard successfully completed.
            "status": "available",

            "fallback_used": (
                fortyguard_temperature is None
                or fortyguard_humidity is None
            ),

            "reason": (
                "Some FortyGuard environmental "
                "parameters were null; Open-Meteo "
                "was used only for missing model inputs."
                if (
                    fortyguard_temperature is None
                    or fortyguard_humidity is None
                )
                else None
            ),

            "activity_id": activity_id,

            "temperature_c": (
                thermal_intelligence.get(
                    "temperature"
                )
            ),

            "humidity_percent": (
                thermal_intelligence.get(
                    "humidity"
                )
            ),

            "heat_index_c": (
                thermal_intelligence.get(
                    "heat_index"
                )
            ),

            "apparent_temperature_c": (
                thermal_intelligence.get(
                    "apparent_temperature"
                )
            ),

            "wet_bulb_temperature_c": (
                thermal_intelligence.get(
                    "wet_bulb_temperature"
                )
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
