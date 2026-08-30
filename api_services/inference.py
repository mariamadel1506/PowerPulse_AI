from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# 1. PATHS / ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Project root:
# api_services/
#     inference.py
# model:
#     power_anomaly_model.pkl
PROJECT_ROOT = BASE_DIR.parent

# Try loading .env locally.
ENV_FILES = [
    BASE_DIR / ".env",
    PROJECT_ROOT / ".env",
    Path.cwd() / ".env",
]

for env_file in ENV_FILES:
    if env_file.exists():
        load_dotenv(
            dotenv_path=env_file,
            override=False,
        )


# ============================================================
# 2. MODEL FILE SEARCH
# ============================================================

MODEL_FILENAME = "power_anomaly_model.pkl"


def find_model_path() -> Path:
    """
    Find the model in several locations.

    This makes the code work both:
    - locally
    - on Vercel
    - when model is in project root
    - when model is next to inference.py
    """

    possible_paths = [
        # Project root
        PROJECT_ROOT / MODEL_FILENAME,

        # Same directory as inference.py
        BASE_DIR / MODEL_FILENAME,

        # Current working directory
        Path.cwd() / MODEL_FILENAME,

        # api directory if present
        PROJECT_ROOT / "api" / MODEL_FILENAME,

        # models directory
        PROJECT_ROOT / "models" / MODEL_FILENAME,

        # api_services/models
        BASE_DIR / "models" / MODEL_FILENAME,
    ]

    checked = []

    for path in possible_paths:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path

        checked.append(str(resolved))

        if resolved.exists() and resolved.is_file():
            print(
                f"[PowerPulse] Model found: {resolved}"
            )
            return resolved

    raise IntegrationError(
        "Model file not found.\n"
        f"Expected filename: {MODEL_FILENAME}\n"
        "Checked:\n"
        + "\n".join(checked)
    )


# ============================================================
# 3. API URLS
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
# 6. ERROR CLASS
# ============================================================

class IntegrationError(Exception):
    pass


# ============================================================
# 7. MODEL CACHE
# ============================================================

_MODEL_CACHE = None


def load_model():
    """
    Load Random Forest model once and cache it.
    """

    global _MODEL_CACHE

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    model_path = find_model_path()

    print(
        f"[PowerPulse] Loading model from: "
        f"{model_path}"
    )

    try:
        _MODEL_CACHE = joblib.load(
            model_path
        )
    except Exception as error:
        raise IntegrationError(
            "Failed to load model "
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

    api_key = os.getenv(
        "FORTYGUARD_API_KEY",
        "",
    ).strip()

    if not api_key:
        raise IntegrationError(
            "FORTYGUARD_API_KEY is missing. "
            "Add it to Vercel Environment Variables."
        )

    return api_key


# ============================================================
# 9. FORTYGUARD HEADERS
# ============================================================

def get_fortyguard_headers() -> Dict[str, str]:

    return {
        "api-key": get_api_key(),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# ============================================================
# 10. LOCATION
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
# 11. OPEN-METEO
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
            timeout=20,
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

    if temperature is None:
        raise IntegrationError(
            "Open-Meteo returned no temperature."
        )

    if humidity is None:
        raise IntegrationError(
            "Open-Meteo returned no humidity."
        )

    if wind_speed is None:
        raise IntegrationError(
            "Open-Meteo returned no wind speed."
        )

    if timestamp is None:
        timestamp = datetime.utcnow().strftime(
            "%Y-%m-%dT%H:%M"
        )

    return {

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


# ============================================================
# 12. FORTYGUARD SUBMIT
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

    except ValueError:

        # Open-Meteo can return a timezone-less
        # local timestamp.
        try:

            timestamp_dt = datetime.strptime(
                str(timestamp),
                "%Y-%m-%dT%H:%M",
            )

        except ValueError as error:

            raise IntegrationError(
                "Invalid weather timestamp: "
                f"{timestamp}"
            ) from error

    payload = {

        "latitude": float(
            latitude
        ),

        "longitude": float(
            longitude
        ),

        "temperature": float(
            temperature
        ),

        "date_time": {

            "start_date": (
                timestamp_dt.strftime(
                    "%Y-%m-%d"
                )
            ),

            "start_time": (
                timestamp_dt.strftime(
                    "%H:%M"
                )
            ),

            "filter_type": 1,
        },
    }

    url = (
        f"{FORTYGUARD_URL}"
        "/env_params"
    )

    print(
        "[FortyGuard] Submitting "
        "environment analysis..."
    )

    try:

        response = requests.post(
            url,
            headers=get_fortyguard_headers(),
            json=payload,
            timeout=20,
        )

    except requests.RequestException as error:

        raise IntegrationError(
            "FortyGuard connection error: "
            f"{error}"
        ) from error

    try:

        response_data = response.json()

    except ValueError:

        raise IntegrationError(
            "FortyGuard returned invalid JSON: "
            f"{response.text[:500]}"
        )

    if response.status_code >= 400:

        raise IntegrationError(
            "FortyGuard submission failed "
            f"(HTTP {response.status_code}): "
            f"{response_data}"
        )

    if response_data.get(
        "error"
    ) is True:

        raise IntegrationError(
            "FortyGuard rejected request: "
            f"{response_data.get('message')}"
        )

    data = response_data.get(
        "data"
    )

    if not isinstance(
        data,
        dict,
    ):

        data = {}

    activity_id = (
        data.get("activity_id")
        or response_data.get(
            "activity_id"
        )
    )

    if not activity_id:

        raise IntegrationError(
            "FortyGuard did not return "
            "activity_id. Response: "
            f"{response_data}"
        )

    activity_id = str(
        activity_id
    )

    print(
        "[FortyGuard] Activity submitted: "
        f"{activity_id}"
    )

    return activity_id


# ============================================================
# 13. FORTYGUARD STATUS
# ============================================================

def get_fortyguard_result(
    activity_id: str,
    max_attempts: int = 12,
    wait_seconds: float = 1.5,
) -> Dict[str, Any]:

    activity_id = str(
        activity_id
    ).strip()

    url = (
        f"{FORTYGUARD_URL}"
        f"/status/{activity_id}"
    )

    last_data: Dict[str, Any] = {}

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        try:

            response = requests.get(
                url,
                headers=get_fortyguard_headers(),
                timeout=15,
            )

        except requests.RequestException as error:

            raise IntegrationError(
                "FortyGuard status connection error: "
                f"{error}"
            ) from error

        try:

            response_data = response.json()

        except ValueError:

            raise IntegrationError(
                "FortyGuard status returned "
                "invalid JSON."
            )

        if response.status_code >= 400:

            raise IntegrationError(
                "FortyGuard status failed "
                f"(HTTP {response.status_code}): "
                f"{response_data}"
            )

        if response_data.get(
            "error"
        ) is True:

            raise IntegrationError(
                "FortyGuard status API error: "
                f"{response_data.get('message')}"
            )

        data = response_data.get(
            "data"
        )

        if not isinstance(
            data,
            dict,
        ):

            data = response_data

        last_data = data

        status = str(
            data.get("status")
            or ""
        ).strip().lower()

        print(
            "[FortyGuard] "
            f"{attempt}/{max_attempts} "
            f"status={status}"
        )

        # ====================================================
        # COMPLETED
        # ====================================================

        if status in {
            "completed",
            "complete",
            "success",
            "succeeded",
            "done",
            "ok",
        }:

            result = data.get(
                "result"
            )

            if not isinstance(
                result,
                dict,
            ):

                result = {}

            data["result"] = result

            return data

        # ====================================================
        # FAILED
        # ====================================================

        if status in {
            "failed",
            "failure",
            "error",
        }:

            message = (
                data.get("message")
                or data.get("error")
                or response_data.get(
                    "message"
                )
                or "FortyGuard processing failed."
            )

            raise IntegrationError(
                f"FortyGuard processing failed: "
                f"{message}"
            )

        # ====================================================
        # PROCESSING
        # ====================================================

        if attempt < max_attempts:

            time.sleep(
                wait_seconds
            )

    # ========================================================
    # IMPORTANT:
    # Do NOT crash the entire analysis.
    #
    # Return Processing so the caller can still use
    # Open-Meteo and the ML model.
    # ========================================================

    return {
        "activity_id": activity_id,
        "status": "Processing",
        "result": {},
        "_poll_timeout": True,
        "_last_data": last_data,
    }


# ============================================================
# 14. SAFE VALUE EXTRACTION
# ============================================================

def _first_non_null_value(
    obj: Dict[str, Any],
    keys: list[str],
) -> Optional[Any]:

    for key in keys:

        if key not in obj:
            continue

        value = obj.get(
            key
        )

        if value is None:
            continue

        # FortyGuard may return:
        #
        # [
        #     None,
        #     72.5,
        #     73.1
        # ]
        #
        # We must NOT blindly use [0].

        if isinstance(
            value,
            list,
        ):

            for item in value:

                if item is not None:
                    return item

            continue

        return value

    return None


# ============================================================
# 15. SAFE FLOAT
# ============================================================

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
# 16. EXTRACT FORTYGUARD
# ============================================================

def extract_fortyguard_intelligence(
    data: Dict[str, Any],
) -> Dict[str, Any]:

    result = data.get(
        "result"
    )

    if not isinstance(
        result,
        dict,
    ):

        result = {}

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

            return {
                "temperature": None,
                "latitude": None,
                "longitude": None,
                "humidity": None,
                "heat_index": None,
                "apparent_temperature": None,
                "wet_bulb_temperature": None,
            }

    location = locations[0]

    if not isinstance(
        location,
        dict,
    ):

        return {
            "temperature": None,
            "latitude": None,
            "longitude": None,
            "humidity": None,
            "heat_index": None,
            "apparent_temperature": None,
            "wet_bulb_temperature": None,
        }

    parameters = location.get(
        "parameters"
    )

    if not isinstance(
        parameters,
        dict,
    ):

        parameters = {}

    # ========================================================
    # LOCATION
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

    apparent_temperature = (
        _first_non_null_value(
            parameters,
            [
                "apparent_temperature_celsius",
                "apparent_temperature",
                "feels_like",
            ],
        )
    )

    # ========================================================
    # WET BULB
    # ========================================================

    wet_bulb_temperature = (
        _first_non_null_value(
            parameters,
            [
                "wet_bulb_temperature_celsius",
                "wet_bulb_temperature",
                "wet_bulb",
            ],
        )
    )

    return {

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


# ============================================================
# 17. MODEL INPUT
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

    ratio = (
        current /
        (baseline + epsilon)
    )

    change_pct = (
        difference /
        (baseline + epsilon)
    ) * 100.0

    interaction = (
        temp * current
    )

    heatwave = int(
        temp > 35.0
        and ratio > 1.10
    )

    return pd.DataFrame(
        [{
            "Electricity_Consumed": current,
            "Temperature": temp,
            "Humidity": hum,
            "Wind_Speed": wind,
            "Avg_Past_Consumption": baseline,
            "Difference": difference,
            "Consumption_Ratio": ratio,
            "Consumption_Change_Percentage": change_pct,
            "Temp_Consumption_Interaction": interaction,
            "Heatwave_Anomaly_Risk": heatwave,
        }],
        columns=FEATURE_ORDER,
    )


# ============================================================
# 18. MODEL PREDICTION
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
    # VERY LOW CONSUMPTION
    # ========================================================

    if current <= 0 or ratio < 0.25:

        return {
            "prediction": 1,
            "prediction_status": "Abnormal",
            "label": "Abnormal",
            "risk_level": "CRITICAL",
            "risk_score": 100.0,
            "action_code": "CRITICAL_INSPECT",
            "action": (
                "Immediate field inspection "
                "is required."
            ),
            "abnormal_probability": 1.0,
            "consumption_ratio": round(
                ratio,
                4,
            ),
        }

    # ========================================================
    # MODEL
    # ========================================================

    try:

        model_prediction = int(
            model.predict(
                model_input
            )[0]
        )

    except Exception as error:

        raise IntegrationError(
            "Model prediction failed: "
            f"{error}"
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

                index = classes.index(
                    1
                )

                probability = float(
                    probabilities[index]
                )

            else:

                probability = (
                    1.0
                    if model_prediction == 1
                    else 0.0
                )

        except Exception as error:

            print(
                "[PowerPulse] Probability "
                f"warning: {error}"
            )

            probability = (
                1.0
                if model_prediction == 1
                else 0.0
            )

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
    # FINAL CLASS
    # ========================================================

    if probability > 0.50:

        prediction = 1
        status = "Potentially Abnormal"
        label = "Abnormal"

    else:

        prediction = 0
        status = "Normal"
        label = "Normal"

    # ========================================================
    # RISK
    # ========================================================

    if (
        ratio < 0.50
        or ratio > 2.00
    ):

        risk_level = "CRITICAL"
        risk_score = 100.0
        action_code = "CRITICAL_INSPECT"

        action = (
            "Urgent field inspection "
            "of the electrical meter "
            "and infrastructure is recommended."
        )

    elif (
        ratio < 0.80
        or ratio > 1.20
    ):

        risk_level = "MEDIUM"
        risk_score = 50.0
        action_code = "INVESTIGATE"

        action = (
            "Conduct remote telemetry "
            "audit and investigate "
            "the consumption deviation."
        )

    else:

        risk_level = "LOW"
        risk_score = 0.0
        action_code = "MONITOR"

        action = (
            "Maintain routine operational monitoring."
        )

    # ========================================================
    # NORMAL OVERRIDE
    # ========================================================

    if prediction == 0:

        risk_level = "LOW"
        risk_score = 0.0
        action_code = "MONITOR"

        action = (
            "Maintain routine operational monitoring."
        )

    return {

        "prediction": prediction,

        "prediction_status": status,

        "label": label,

        "risk_level": risk_level,

        "risk_score": risk_score,

        "action_code": action_code,

        "action": action,

        "abnormal_probability": probability,

        "consumption_ratio": round(
            ratio,
            4,
        ),
    }


# ============================================================
# 19. COMPLETE ANALYSIS
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
    # LOCATION
    # ========================================================

    latitude, longitude = (
        get_state_coordinates(
            state
        )
    )

    # ========================================================
    # WEATHER
    # ========================================================

    weather = get_current_weather(
        latitude,
        longitude,
    )

    # ========================================================
    # FORTYGUARD
    # ========================================================

    fortyguard_status = "unavailable"
    fortyguard_activity_id = None

    thermal = {
        "temperature": None,
        "latitude": None,
        "longitude": None,
        "humidity": None,
        "heat_index": None,
        "apparent_temperature": None,
        "wet_bulb_temperature": None,
    }

    fortyguard_reason = None

    try:

        print(
            "[PowerPulse] Starting "
            "FortyGuard..."
        )

        fortyguard_activity_id = (
            submit_fortyguard(
                latitude=latitude,
                longitude=longitude,
                temperature=weather[
                    "temperature"
                ],
                timestamp=weather[
                    "timestamp"
                ],
            )
        )

        fortyguard_data = (
            get_fortyguard_result(
                activity_id=(
                    fortyguard_activity_id
                ),

                # Keep this bounded for Vercel.
                max_attempts=12,

                wait_seconds=1.5,
            )
        )

        fortyguard_status = str(
            fortyguard_data.get(
                "status",
                "Processing",
            )
        )

        thermal = (
            extract_fortyguard_intelligence(
                fortyguard_data
            )
        )

        if (
            fortyguard_status.lower()
            in {
                "completed",
                "complete",
                "success",
                "succeeded",
                "done",
                "ok",
            }
        ):

            fortyguard_status = (
                "available"
            )

            print(
                "[PowerPulse] "
                "FortyGuard completed."
            )

        else:

            fortyguard_status = (
                "processing"
            )

            fortyguard_reason = (
                "FortyGuard activity is "
                "still processing."
            )

    except Exception as error:

        # ====================================================
        # IMPORTANT
        #
        # FortyGuard must NOT kill the whole ML analysis.
        # ====================================================

        fortyguard_status = (
            "unavailable"
        )

        fortyguard_reason = str(
            error
        )

        print(
            "[PowerPulse] "
            f"FortyGuard warning: {error}"
        )

    # ========================================================
    # FORTYGUARD FALLBACKS
    #
    # These do NOT pretend Open-Meteo is FortyGuard.
    #
    # We only use Open-Meteo for missing values required
    # by the ML model.
    # ========================================================

    final_temperature = (
        thermal["temperature"]
    )

    final_humidity = (
        thermal["humidity"]
    )

    if final_temperature is None:

        final_temperature = (
            weather["temperature"]
        )

        if fortyguard_status == "available":

            fortyguard_reason = (
                "FortyGuard completed, "
                "but temperature was missing; "
                "Open-Meteo used only for model input."
            )

    if final_humidity is None:

        final_humidity = (
            weather["humidity"]
        )

        if fortyguard_status == "available":

            fortyguard_reason = (
                "FortyGuard completed, "
                "but relative humidity was missing; "
                "Open-Meteo used only for model input."
            )

    # ========================================================
    # MODEL
    # ========================================================

    model_input = create_model_input(
        current_consumption=(
            current_consumption
        ),

        avg_past_consumption=(
            avg_past_consumption
        ),

        temperature=float(
            final_temperature
        ),

        humidity=float(
            final_humidity
        ),

        wind_speed=float(
            weather["wind_speed"]
        ),
    )

    model = load_model()

    prediction = predict_anomaly(
        model=model,
        model_input=model_input,
        current_consumption=(
            current_consumption
        ),
        avg_past_consumption=(
            avg_past_consumption
        ),
    )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "location": {

            "state": state,

            "latitude": float(
                latitude
            ),

            "longitude": float(
                longitude
            ),
        },

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

        # ====================================================
        # FORTYGUARD
        # ====================================================

        "fortyguard": {

            "status": fortyguard_status,

            "fallback_used": (
                fortyguard_status
                != "available"
                or (
                    thermal[
                        "temperature"
                    ]
                    is None
                    or
                    thermal[
                        "humidity"
                    ]
                    is None
                )
            ),

            "reason": fortyguard_reason,

            "activity_id": (
                fortyguard_activity_id
            ),

            # These values are ONLY actual FortyGuard
            # values. We do not overwrite them with weather.

            "temperature_c": (
                thermal[
                    "temperature"
                ]
            ),

            "humidity_percent": (
                thermal[
                    "humidity"
                ]
            ),

            "heat_index_c": (
                thermal[
                    "heat_index"
                ]
            ),

            "apparent_temperature_c": (
                thermal[
                    "apparent_temperature"
                ]
            ),

            "wet_bulb_temperature_c": (
                thermal[
                    "wet_bulb_temperature"
                ]
            ),

            # =================================================
            # Helpful UI/model fallback fields
            # =================================================

            "model_temperature_c": float(
                final_temperature
            ),

            "model_humidity_percent": float(
                final_humidity
            ),
        },

        # ====================================================
        # CONSUMPTION
        # ====================================================

        "consumption": {

            "current_kwh": (
                current_consumption
            ),

            "historical_baseline_kwh": (
                avg_past_consumption
            ),
        },

        # ====================================================
        # MODEL
        # ====================================================

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


# ============================================================
# 20. OPTIONAL LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print(
        "======================================"
    )

    print(
        "PowerPulse inference.py loaded."
    )

    print(
        f"BASE_DIR: {BASE_DIR}"
    )

    print(
        f"PROJECT_ROOT: {PROJECT_ROOT}"
    )

    print(
        f"FORTYGUARD_URL: {FORTYGUARD_URL}"
    )

    try:

        model_path = find_model_path()

        print(
            f"MODEL: {model_path}"
        )

    except Exception as error:

        print(
            f"MODEL ERROR: {error}"
        )

    print(
        "======================================"
    )
