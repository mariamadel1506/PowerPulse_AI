from datetime import datetime
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
import joblib
import pandas as pd
import requests


# ============================================================
# 1. PATHS / ENVIRONMENT
# ============================================================

# inference.py:
# project/
# ├── main.py
# ├── power_anomaly_model.pkl
# ├── .env
# └── api_services/
#     └── inference.py

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Load .env from both possible locations.
load_dotenv(PROJECT_ROOT / ".env", override=False)
load_dotenv(BASE_DIR / ".env", override=False)


MODEL_FILENAME = "power_anomaly_model.pkl"

# Search several locations so it works locally AND on Vercel.
MODEL_CANDIDATES = [
    PROJECT_ROOT / MODEL_FILENAME,
    BASE_DIR / MODEL_FILENAME,
    Path.cwd() / MODEL_FILENAME,
]


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

FORTYGUARD_URL = os.getenv(
    "FORTYGUARD_BASE_URL",
    "https://api.fortyguard.com/v1",
).strip().rstrip("/")


# ============================================================
# 2. MODEL FEATURES
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
# 3. US STATE COORDINATES
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
    "Rhode Island": (41.8240, -71.4121),
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
# 4. CUSTOM ERROR
# ============================================================

class IntegrationError(Exception):
    pass


# ============================================================
# 5. MODEL LOADING
# ============================================================

def load_model():
    """
    Load the trained RandomForest model.

    Works both locally and on Vercel by searching:
        1. project root
        2. api_services/
        3. current working directory
    """

    existing_paths = []

    for candidate in MODEL_CANDIDATES:
        try:
            if candidate.exists() and candidate.is_file():
                existing_paths.append(candidate)
        except Exception:
            continue

    if not existing_paths:
        raise IntegrationError(
            "Model file not found. Searched:\n"
            + "\n".join(
                f"- {path}"
                for path in MODEL_CANDIDATES
            )
        )

    model_path = existing_paths[0]

    print(
        f"[MODEL] Loading model from: {model_path}"
    )

    try:
        model = joblib.load(model_path)

    except Exception as error:
        raise IntegrationError(
            f"Failed to load model '{model_path}': {error}"
        ) from error

    print(
        "[MODEL] Model loaded successfully."
    )

    return model


# ============================================================
# 6. FORTYGUARD CONFIGURATION
# ============================================================

def get_api_key() -> str:

    api_key = os.getenv(
        "FORTYGUARD_API_KEY",
        "",
    ).strip()

    if not api_key:

        raise IntegrationError(
            "FORTYGUARD_API_KEY is missing. "
            "Add FORTYGUARD_API_KEY to Vercel Environment Variables."
        )

    return api_key


def get_fortyguard_headers() -> Dict[str, str]:

    return {
        "api-key": get_api_key(),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# ============================================================
# 7. LOCATION
# ============================================================

def get_state_coordinates(
    state: str,
) -> Tuple[float, float]:

    normalized_state = str(
        state
    ).strip()

    if normalized_state not in STATE_COORDINATES:

        raise IntegrationError(
            f"Invalid US state: {normalized_state}"
        )

    return STATE_COORDINATES[
        normalized_state
    ]


# ============================================================
# 8. OPEN-METEO
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
        "[Weather] Requesting current weather..."
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

    if response.status_code >= 400:

        raise IntegrationError(
            f"Weather API error "
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
            "Weather temperature is missing."
        )

    if humidity is None:
        raise IntegrationError(
            "Weather humidity is missing."
        )

    if wind_speed is None:
        raise IntegrationError(
            "Weather wind speed is missing."
        )

    if timestamp is None:
        raise IntegrationError(
            "Weather timestamp is missing."
        )

    result = {
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
        "[Weather] "
        f"T={result['temperature']}°C "
        f"H={result['humidity']}% "
        f"W={result['wind_speed']} km/h"
    )

    return result


# ============================================================
# 9. FORTYGUARD SUBMISSION
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
            "start_date": timestamp_dt.strftime(
                "%Y-%m-%d"
            ),
            "start_time": timestamp_dt.strftime(
                "%H:%M"
            ),
            "filter_type": 1,
        },
    }

    url = (
        f"{FORTYGUARD_URL}"
        "/env_params"
    )

    print(
        "[FortyGuard] =================================="
    )

    print(
        "[FortyGuard] SUBMIT"
    )

    print(
        f"[FortyGuard] URL: {url}"
    )

    print(
        f"[FortyGuard] payload: {payload}"
    )

    try:

        response = requests.post(
            url,
            headers=get_fortyguard_headers(),
            json=payload,
            timeout=30,
        )

    except requests.RequestException as error:

        raise IntegrationError(
            f"FortyGuard connection error: {error}"
        ) from error

    try:

        response_data = response.json()

    except ValueError:

        response_data = None

    print(
        "[FortyGuard] HTTP:",
        response.status_code,
    )

    print(
        "[FortyGuard] response:",
        response_data
        if response_data is not None
        else response.text,
    )

    if response.status_code >= 400:

        raise IntegrationError(
            "FortyGuard submission failed "
            f"(HTTP {response.status_code}): "
            f"{response_data or response.text}"
        )

    if not isinstance(
        response_data,
        dict,
    ):

        raise IntegrationError(
            "FortyGuard returned invalid JSON."
        )

    if response_data.get(
        "error"
    ) is True:

        raise IntegrationError(
            "FortyGuard rejected the request: "
            f"{response_data.get('message', response_data)}"
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
        or response_data.get("activity_id")
    )

    if not activity_id:

        raise IntegrationError(
            "FortyGuard did not return activity_id. "
            f"Response: {response_data}"
        )

    activity_id = str(
        activity_id
    ).strip()

    print(
        "[FortyGuard] Activity ID:",
        activity_id,
    )

    print(
        "[FortyGuard] =================================="
    )

    return activity_id


# ============================================================
# 10. FORTYGUARD POLLING
# ============================================================

def get_fortyguard_result(
    activity_id: str,
    max_attempts: int = 30,
    wait_seconds: float = 1.0,
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
        "[FortyGuard] Starting polling..."
    )

    last_response = None

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

            print(
                f"[FortyGuard] Poll connection error "
                f"on attempt {attempt}: {error}"
            )

            if attempt < max_attempts:

                time.sleep(
                    wait_seconds
                )

                continue

            raise IntegrationError(
                f"FortyGuard polling failed: {error}"
            ) from error

        last_response = response

        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

        try:

            response_data = response.json()

        except ValueError:

            print(
                "[FortyGuard] Invalid JSON on polling."
            )

            if attempt < max_attempts:

                time.sleep(
                    wait_seconds
                )

                continue

            raise IntegrationError(
                "FortyGuard status returned invalid JSON. "
                f"HTTP {response.status_code}: "
                f"{response.text}"
            )

        print(
            f"[FortyGuard] Poll "
            f"{attempt}/{max_attempts} "
            f"HTTP={response.status_code}"
        )

        # ----------------------------------------------------
        # 404
        #
        # Sometimes the activity needs a short moment
        # before the status endpoint recognizes it.
        # ----------------------------------------------------

        if response.status_code == 404:

            print(
                "[FortyGuard] Activity not visible yet."
            )

            if attempt < max_attempts:

                time.sleep(
                    wait_seconds
                )

                continue

            raise IntegrationError(
                "FortyGuard activity was not found after "
                f"{max_attempts} polling attempts. "
                f"activity_id={activity_id}"
            )

        # ----------------------------------------------------
        # Other HTTP errors
        # ----------------------------------------------------

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

        if response_data.get(
            "error"
        ) is True:

            raise IntegrationError(
                "FortyGuard status API error: "
                f"{response_data.get('message', response_data)}"
            )

        data = response_data.get(
            "data"
        )

        if not isinstance(
            data,
            dict,
        ):

            data = response_data

        status = str(
            data.get("status")
            or response_data.get("status")
            or ""
        ).strip().lower()

        print(
            "[FortyGuard] status =",
            status,
        )

        # ----------------------------------------------------
        # COMPLETED
        # ----------------------------------------------------

        if status in {
            "completed",
            "complete",
            "success",
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

                raise IntegrationError(
                    "FortyGuard says Completed "
                    "but result is missing."
                )

            print(
                "[FortyGuard] COMPLETED SUCCESSFULLY."
            )

            print(
                "[FortyGuard] Result keys:",
                list(result.keys()),
            )

            return data

        # ----------------------------------------------------
        # FAILED
        # ----------------------------------------------------

        if status in {
            "failed",
            "failure",
            "error",
            "cancelled",
            "canceled",
        }:

            message = (
                data.get("message")
                or data.get("error")
                or response_data.get("message")
                or "FortyGuard activity failed."
            )

            raise IntegrationError(
                f"FortyGuard processing error: "
                f"{message}"
            )

        # ----------------------------------------------------
        # PROCESSING
        # ----------------------------------------------------

        if attempt < max_attempts:

            time.sleep(
                wait_seconds
            )

    raise TimeoutError(
        "FortyGuard activity did not complete within "
        f"{max_attempts * wait_seconds} seconds. "
        f"activity_id={activity_id}. "
        "The activity may still be processing on FortyGuard."
    )


# ============================================================
# 11. VALUE EXTRACTION
# ============================================================

def first_non_null(
    obj: Dict[str, Any],
    keys: list[str],
) -> Optional[Any]:

    """
    VERY IMPORTANT:

    FortyGuard parameters are arrays.

    Example:
        [None, 19.8]

    The old code used [0] and returned None.

    This function searches the entire array and returns
    the first actual value.
    """

    for key in keys:

        if key not in obj:
            continue

        value = obj.get(
            key
        )

        if value is None:
            continue

        if isinstance(
            value,
            list,
        ):

            for item in value:

                if item is None:
                    continue

                if item == "":
                    continue

                return item

            continue

        return value

    return None


# ============================================================
# 12. FORTYGUARD EXTRACTION
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

    # --------------------------------------------------------
    # Find the first valid location
    # --------------------------------------------------------

    location = None

    for item in locations:

        if isinstance(
            item,
            dict,
        ):

            location = item
            break

    if location is None:

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

    print(
        "[FortyGuard] Available parameter keys:"
    )

    print(
        list(
            parameters.keys()
        )
    )

    # --------------------------------------------------------
    # Location values
    # --------------------------------------------------------

    latitude = first_non_null(
        location,
        [
            "lat",
            "latitude",
        ],
    )

    longitude = first_non_null(
        location,
        [
            "lon",
            "longitude",
        ],
    )

    temperature = first_non_null(
        location,
        [
            "temperature",
            "temp",
            "temperature_celsius",
        ],
    )

    if temperature is None:

        temperature = first_non_null(
            parameters,
            [
                "temperature",
                "temp",
                "temperature_celsius",
            ],
        )

    # --------------------------------------------------------
    # Humidity
    # --------------------------------------------------------

    humidity = first_non_null(
        parameters,
        [
            "relative_humidity_percent",
            "relative_humidity",
            "humidity",
        ],
    )

    # --------------------------------------------------------
    # Thermal metrics
    # --------------------------------------------------------

    heat_index = first_non_null(
        parameters,
        [
            "heat_index_celsius",
            "heat_index",
        ],
    )

    apparent_temperature = first_non_null(
        parameters,
        [
            "apparent_temperature_celsius",
            "apparent_temperature",
            "feels_like",
        ],
    )

    wet_bulb_temperature = first_non_null(
        parameters,
        [
            "wet_bulb_temperature_celsius",
            "wet_bulb_temperature",
            "wet_bulb",
        ],
    )

    def safe_float(
        value
    ):

        if value is None:
            return None

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    extracted = {
        "temperature": safe_float(
            temperature
        ),

        "latitude": safe_float(
            latitude
        ),

        "longitude": safe_float(
            longitude
        ),

        "humidity": safe_float(
            humidity
        ),

        "heat_index": safe_float(
            heat_index
        ),

        "apparent_temperature": safe_float(
            apparent_temperature
        ),

        "wet_bulb_temperature": safe_float(
            wet_bulb_temperature
        ),
    }

    print(
        "[FortyGuard] Extracted:"
    )

    print(
        extracted
    )

    return extracted


# ============================================================
# 13. MODEL FEATURE ENGINEERING
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
        current
        - baseline
    )

    consumption_ratio = (
        current
        /
        (
            baseline
            + epsilon
        )
    )

    consumption_change_percentage = (
        difference
        /
        (
            baseline
            + epsilon
        )
    ) * 100.0

    temperature_consumption_interaction = (
        temp
        * current
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
        current
        /
        (
            baseline
            + epsilon
        )
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba",
    ):

        try:

            probabilities = model.predict_proba(
                model_input
            )[0]

            classes = list(
                getattr(
                    model,
                    "classes_",
                    [0, 1],
                )
            )

            if 1 in classes:

                abnormal_index = classes.index(
                    1
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

    # --------------------------------------------------------
    # Final prediction
    # --------------------------------------------------------

    if probability > 0.50:

        final_prediction = 1

        prediction_status = (
            "Potentially Abnormal"
        )

        label = "Abnormal"

    else:

        final_prediction = 0

        prediction_status = "Normal"

        label = "Normal"

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    if (
        current <= 0.0
        or ratio < 0.25
    ):

        risk_level = "CRITICAL"
        risk_score = 100.0
        action_code = "CRITICAL_INSPECT"

        action = (
            "Immediate field inspection required "
            "due to extreme under-consumption."
        )

    elif (
        ratio > 2.00
        or ratio < 0.50
    ):

        risk_level = "HIGH"
        risk_score = 75.0
        action_code = "INSPECT"

        action = (
            "Mandatory dispatch of a field audit "
            "team to conduct physical meter examination."
        )

    elif (
        ratio > 1.20
        or ratio < 0.80
    ):

        risk_level = "MEDIUM"
        risk_score = 50.0
        action_code = "INVESTIGATE"

        action = (
            "Conduct remote telemetry audits of smart "
            "meter logs against localized weather data."
        )

    else:

        risk_level = "LOW"
        risk_score = 0.0
        action_code = "MONITOR"

        action = (
            "Maintain standard operational monitoring schedule."
        )

    # --------------------------------------------------------
    # Normal override
    # --------------------------------------------------------

    if final_prediction == 0:

        prediction_status = "Normal"
        label = "Normal"

        if (
            0.80 <= ratio <= 1.20
        ):

            risk_level = "LOW"
            risk_score = 0.0
            action_code = "MONITOR"

            action = (
                "Maintain standard operational monitoring schedule."
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

    print(
        "[PowerPulse] Starting FortyGuard..."
    )

    activity_id = submit_fortyguard(
        latitude=latitude,
        longitude=longitude,
        temperature=weather[
            "temperature"
        ],
        timestamp=weather[
            "timestamp"
        ],
    )

    fortyguard_data = get_fortyguard_result(
        activity_id=activity_id,
        max_attempts=30,
        wait_seconds=1.0,
    )

    thermal_intelligence = (
        extract_fortyguard_intelligence(
            fortyguard_data
        )
    )

    # ========================================================
    # IMPORTANT:
    # FortyGuard humidity is NOT mandatory for the UI.
    #
    # If FortyGuard doesn't provide humidity, use the
    # weather API humidity ONLY for the ML feature.
    #
    # We do NOT fake FortyGuard humidity.
    # ========================================================

    fg_temperature = (
        thermal_intelligence[
            "temperature"
        ]
    )

    fg_humidity = (
        thermal_intelligence[
            "humidity"
        ]
    )

    if fg_temperature is None:

        # Use Open-Meteo temperature for model input only.
        fg_temperature = float(
            weather["temperature"]
        )

    model_humidity = (
        fg_humidity
        if fg_humidity is not None
        else float(
            weather["humidity"]
        )
    )

    # ========================================================
    # MODEL INPUT
    # ========================================================

    model_input = create_model_input(
        current_consumption=current_consumption,

        avg_past_consumption=avg_past_consumption,

        temperature=fg_temperature,

        humidity=model_humidity,

        wind_speed=weather[
            "wind_speed"
        ],
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = load_model()

    prediction = predict_anomaly(
        model=model,

        model_input=model_input,

        current_consumption=current_consumption,

        avg_past_consumption=avg_past_consumption,
    )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "location": {

            "state": state,

            "latitude": latitude,

            "longitude": longitude,
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

        "fortyguard": {

            # This is TRUE only because we reached
            # FortyGuard Completed successfully.
            "status": "available",

            "fallback_used": False,

            "reason": None,

            "activity_id": activity_id,

            "temperature_c": (
                thermal_intelligence[
                    "temperature"
                ]
            ),

            "humidity_percent": (
                thermal_intelligence[
                    "humidity"
                ]
            ),

            "heat_index_c": (
                thermal_intelligence[
                    "heat_index"
                ]
            ),

            "apparent_temperature_c": (
                thermal_intelligence[
                    "apparent_temperature"
                ]
            ),

            "wet_bulb_temperature_c": (
                thermal_intelligence[
                    "wet_bulb_temperature"
                ]
            ),
        },

        "consumption": {

            "current_kwh": (
                current_consumption
            ),

            "historical_baseline_kwh": (
                avg_past_consumption
            ),
        },

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
