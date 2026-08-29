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
# 1. PATHS / ENVIRONMENT
# ============================================================

# هذا الملف موجود داخل:
# project/
#   main.py
#   generate_html.py
#   power_anomaly_model.pkl
#   api_services/
#       inference.py

INFERENCE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = INFERENCE_DIR.parent

# نحاول تحميل .env من أكثر من مكان
ENV_CANDIDATES = [
    INFERENCE_DIR / ".env",
    PROJECT_DIR / ".env",
]

for env_file in ENV_CANDIDATES:
    if env_file.exists():
        load_dotenv(
            dotenv_path=env_file,
            override=True,
        )

# ------------------------------------------------------------
# Model filenames
#
# ندعم الاسم القديم الذي فيه مسافة
# والاسم الصحيح بدون مسافة
# ------------------------------------------------------------

MODEL_CANDIDATES = [
    INFERENCE_DIR / "power_anomaly_model.pkl",
    INFERENCE_DIR / "power _anomaly_model.pkl",

    PROJECT_DIR / "power_anomaly_model.pkl",
    PROJECT_DIR / "power _anomaly_model.pkl",
]


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

FORTYGUARD_URL = os.getenv(
    "FORTYGUARD_BASE_URL",
    "https://api.fortyguard.com/v1",
).strip().rstrip("/")


# ============================================================
# 2. FEATURE ORDER
# ============================================================

# مهم جدًا:
# لازم يكون نفس الترتيب المستخدم أثناء التدريب.

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
    "New Hampshire": (42.9958, -71.4548),
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
# 4. CUSTOM ERROR
# ============================================================

class IntegrationError(Exception):
    pass


# ============================================================
# 5. MODEL LOADING
# ============================================================

_MODEL = None
_MODEL_PATH_USED: Optional[Path] = None


def find_model_path() -> Path:
    """
    يبحث عن ملف الموديل في الأماكن الصحيحة.

    يدعم:
        power_anomaly_model.pkl
        power _anomaly_model.pkl
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
        "Searched in:\n"
        f"{searched_paths}\n\n"
        "Please make sure the .pkl file is uploaded "
        "with the deployed application."
    )


def load_model():
    """
    Load the trained RandomForest model.

    يتم تحميل الموديل مرة واحدة فقط.
    """

    global _MODEL
    global _MODEL_PATH_USED

    if _MODEL is not None:
        return _MODEL

    model_path = find_model_path()

    try:
        _MODEL = joblib.load(model_path)
        _MODEL_PATH_USED = model_path

        print("=" * 60)
        print("[MODEL] Model loaded successfully")
        print(f"[MODEL] Path: {model_path}")
        print(f"[MODEL] Type: {type(_MODEL).__name__}")

        if hasattr(_MODEL, "n_features_in_"):
            print(
                f"[MODEL] Expected features: "
                f"{_MODEL.n_features_in_}"
            )

        if hasattr(_MODEL, "feature_names_in_"):
            print(
                "[MODEL] Feature names:"
            )
            print(
                list(_MODEL.feature_names_in_)
            )

        print("=" * 60)

        return _MODEL

    except Exception as error:
        raise IntegrationError(
            f"Failed to load model from "
            f"'{model_path}': {error}"
        ) from error


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
            f"Expected it in .env file. "
            f"Checked: {ENV_CANDIDATES}"
        )

    return api_key


def get_fortyguard_headers() -> Dict[str, str]:

    return {
        "api-key": get_api_key(),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# ============================================================
# 7. STATE COORDINATES
# ============================================================

def get_state_coordinates(
    state: str,
) -> tuple[float, float]:

    state = str(state).strip()

    # محاولة مطابقة case-insensitive
    for state_name, coordinates in STATE_COORDINATES.items():

        if state_name.lower() == state.lower():
            return coordinates

    raise IntegrationError(
        f"Invalid US state: {state}"
    )


# ============================================================
# 8. OPEN-METEO WEATHER
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

    timestamp = current.get("time")

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
        "latitude": float(latitude),
        "longitude": float(longitude),
        "temperature": float(temperature),
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

    url = f"{FORTYGUARD_URL}/env_params"

    print(
        "[FortyGuard] submitting "
        "environment analysis..."
    )

    print(
        f"[FortyGuard] URL: {url}"
    )

    print(
        f"[FortyGuard] Coordinates: "
        f"{latitude}, {longitude}"
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
            f"(HTTP {response.status_code}): {body}"
        )

    if not isinstance(
        response_data,
        dict,
    ):

        raise IntegrationError(
            "FortyGuard returned an unexpected "
            f"response: {response.text}"
        )

    if response_data.get("error") is True:

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
        "[FortyGuard] activity submitted "
        f"successfully: {activity_id}"
    )

    return activity_id


# ============================================================
# 10. FORTYGUARD STATUS / POLLING
# ============================================================

def get_fortyguard_result(
    activity_id: str,
    max_attempts: int = 60,
    wait_seconds: int = 2,
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
        "[FortyGuard] polling activity: "
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
                timeout=60,
            )

        except requests.RequestException as error:

            raise IntegrationError(
                "FortyGuard status connection "
                f"error: {error}"
            ) from error

        try:

            response_data = response.json()

        except ValueError as error:

            raise IntegrationError(
                "FortyGuard status returned "
                "invalid JSON. "
                f"HTTP {response.status_code}: "
                f"{response.text}"
            ) from error

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
                "FortyGuard status response "
                "is not an object."
            )

        if response_data.get("error") is True:

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
            f"[FortyGuard] attempt "
            f"{attempt}/{max_attempts} "
            f"status={status}"
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
                    "FortyGuard returned Completed "
                    "but result object is missing."
                )

            return data

        # ----------------------------------------------------
        # FAILED
        # ----------------------------------------------------

        if status in {
            "failed",
            "failure",
            "error",
        }:

            error_message = (
                data.get("message")
                or data.get("error")
                or response_data.get("message")
                or "FortyGuard processing failed."
            )

            raise IntegrationError(
                f"FortyGuard processing error: "
                f"{error_message}"
            )

        # ----------------------------------------------------
        # PROCESSING
        # ----------------------------------------------------

        if attempt < max_attempts:

            time.sleep(
                wait_seconds
            )

    raise TimeoutError(
        "FortyGuard activity did not complete "
        f"within {max_attempts * wait_seconds} seconds. "
        f"activity_id={activity_id}"
    )


# ============================================================
# 11. SAFE VALUE EXTRACTION
# ============================================================

def _first_value(
    obj: Dict[str, Any],
    keys: list[str],
):

    if not isinstance(
        obj,
        dict,
    ):
        return None

    for key in keys:

        if key not in obj:
            continue

        value = obj.get(key)

        if value is None:
            continue

        if isinstance(
            value,
            list,
        ):

            if not value:
                continue

            value = value[0]

        return value

    return None


def _to_float_or_none(value):

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
# 12. EXTRACT FORTYGUARD INTELLIGENCE
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

    temperature = _first_value(
        location,
        [
            "temperature",
            "temp",
            "temperature_celsius",
        ],
    )

    if temperature is None:

        temperature = _first_value(
            parameters,
            [
                "temperature",
                "temp",
                "temperature_celsius",
            ],
        )

    latitude = _first_value(
        location,
        [
            "lat",
            "latitude",
        ],
    )

    longitude = _first_value(
        location,
        [
            "lon",
            "longitude",
        ],
    )

    humidity = _first_value(
        parameters,
        [
            "relative_humidity_percent",
            "relative_humidity",
            "humidity",
        ],
    )

    heat_index = _first_value(
        parameters,
        [
            "heat_index_celsius",
            "heat_index",
        ],
    )

    apparent_temperature = _first_value(
        parameters,
        [
            "apparent_temperature_celsius",
            "apparent_temperature",
            "feels_like",
        ],
    )

    wet_bulb_temperature = _first_value(
        parameters,
        [
            "wet_bulb_temperature_celsius",
            "wet_bulb_temperature",
            "wet_bulb",
        ],
    )

    return {

        "temperature":
            _to_float_or_none(
                temperature
            ),

        "latitude":
            _to_float_or_none(
                latitude
            ),

        "longitude":
            _to_float_or_none(
                longitude
            ),

        "humidity":
            _to_float_or_none(
                humidity
            ),

        "heat_index":
            _to_float_or_none(
                heat_index
            ),

        "apparent_temperature":
            _to_float_or_none(
                apparent_temperature
            ),

        "wet_bulb_temperature":
            _to_float_or_none(
                wet_bulb_temperature
            ),
    }


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
        / (baseline + epsilon)
    )

    consumption_change_percentage = (
        difference
        / (baseline + epsilon)
    ) * 100.0

    temperature_consumption_interaction = (
        temp * current
    )

    heatwave_anomaly_risk = int(
        temp > 35.0
        and consumption_ratio > 1.10
    )

    input_data = {

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

    df = pd.DataFrame(
        input_data,
        columns=FEATURE_ORDER,
    )

    return df


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
        / (baseline + epsilon)
    )

    # --------------------------------------------------------
    # Validate feature count
    # --------------------------------------------------------

    if hasattr(
        model,
        "n_features_in_",
    ):

        expected = int(
            model.n_features_in_
        )

        actual = len(
            FEATURE_ORDER
        )

        if expected != actual:

            raise IntegrationError(
                "Model feature count mismatch. "
                f"Model expects {expected}, "
                f"but inference provides {actual}."
            )

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    try:

        raw_prediction = model.predict(
            model_input
        )[0]

        model_prediction = int(
            raw_prediction
        )

    except Exception as error:

        raise IntegrationError(
            f"Model prediction failed: {error}"
        ) from error

    # --------------------------------------------------------
    # MODEL PROBABILITY
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

    # --------------------------------------------------------
    # FINAL MODEL LABEL
    # --------------------------------------------------------

    if model_prediction == 1:

        prediction_status = (
            "Potentially Abnormal"
        )

        label = "Abnormal"

    else:

        prediction_status = "Normal"

        label = "Normal"

    # --------------------------------------------------------
    # RISK BASED ON MODEL + CONSUMPTION DEVIATION
    # --------------------------------------------------------

    if model_prediction == 1:

        if (
            ratio < 0.25
            or ratio > 2.00
        ):

            risk_level = "CRITICAL"
            risk_score = 100.0
            action_code = "CRITICAL_INSPECT"

            action = (
                "Urgent field inspection required "
                "due to a severe consumption anomaly."
            )

        elif (
            ratio < 0.50
            or ratio > 1.70
        ):

            risk_level = "HIGH"
            risk_score = 75.0
            action_code = "INSPECT"

            action = (
                "Dispatch a field audit team to "
                "inspect the smart meter and "
                "associated electrical infrastructure."
            )

        else:

            risk_level = "MEDIUM"
            risk_score = 50.0
            action_code = "INVESTIGATE"

            action = (
                "Conduct remote telemetry audits "
                "and investigate the detected "
                "consumption anomaly."
            )

    else:

        risk_level = "LOW"
        risk_score = 0.0
        action_code = "MONITOR"

        action = (
            "Maintain routine operational monitoring."
        )

    return {

        "prediction": model_prediction,

        "prediction_status":
            prediction_status,

        "label":
            label,

        "risk_level":
            risk_level,

        "risk_score":
            risk_score,

        "action":
            action,

        "action_code":
            action_code,

        "abnormal_probability":
            probability,

        "consumption_ratio":
            round(
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
    # 1. LOAD MODEL FIRST
    # ========================================================

    # مهم:
    # لو الموديل مش موجود، هنكتشف المشكلة فورًا
    # بدل ما ننتظر FortyGuard.

    model = load_model()

    # ========================================================
    # 2. LOCATION
    # ========================================================

    latitude, longitude = (
        get_state_coordinates(
            state
        )
    )

    # ========================================================
    # 3. WEATHER
    # ========================================================

    weather = get_current_weather(
        latitude,
        longitude,
    )

    # ========================================================
    # 4. FORTYGUARD
    # ========================================================

    print(
        "[PowerPulse] Starting "
        "FortyGuard integration..."
    )

    activity_id = submit_fortyguard(
        latitude=latitude,
        longitude=longitude,
        temperature=weather["temperature"],
        timestamp=weather["timestamp"],
    )

    fortyguard_data = get_fortyguard_result(
        activity_id=activity_id,
        max_attempts=60,
        wait_seconds=2,
    )

    thermal_intelligence = (
        extract_fortyguard_intelligence(
            fortyguard_data
        )
    )

    print(
        "[PowerPulse] FortyGuard completed successfully."
    )

    # ========================================================
    # 5. GET REAL FORTYGUARD VALUES
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

    # --------------------------------------------------------
    # Important:
    # If FortyGuard does not return these two values,
    # we do NOT silently fake them.
    # --------------------------------------------------------

    if fg_temperature is None:

        raise IntegrationError(
            "FortyGuard completed but did not return "
            "temperature data."
        )

    if fg_humidity is None:

        raise IntegrationError(
            "FortyGuard completed but did not return "
            "relative humidity data."
        )

    # ========================================================
    # 6. MODEL INPUT
    # ========================================================

    model_input = create_model_input(
        current_consumption=current_consumption,
        avg_past_consumption=avg_past_consumption,
        temperature=fg_temperature,
        humidity=fg_humidity,
        wind_speed=weather["wind_speed"],
    )

    print(
        "[MODEL] Input features:"
    )

    print(
        model_input.to_dict(
            orient="records"
        )[0]
    )

    # ========================================================
    # 7. MODEL PREDICTION
    # ========================================================

    prediction = predict_anomaly(
        model=model,
        model_input=model_input,
        current_consumption=current_consumption,
        avg_past_consumption=avg_past_consumption,
    )

    print(
        "[MODEL] Prediction: "
        f"{prediction['label']}"
    )

    print(
        "[MODEL] Probability: "
        f"{prediction['abnormal_probability']}"
    )

    print(
        "[MODEL] Ratio: "
        f"{prediction['consumption_ratio']}"
    )

    # ========================================================
    # 8. FINAL RESPONSE FOR UI
    # ========================================================

    return {

        "location": {

            "state": state,

            "latitude":
                latitude,

            "longitude":
                longitude,
        },

        "weather": {

            "temperature_c":
                float(
                    weather[
                        "temperature"
                    ]
                ),

            "humidity_percent":
                float(
                    weather[
                        "humidity"
                    ]
                ),

            "wind_speed_kmh":
                float(
                    weather[
                        "wind_speed"
                    ]
                ),

            "timestamp":
                weather[
                    "timestamp"
                ],
        },

        # ====================================================
        # FORTYGUARD
        # ====================================================

        "fortyguard": {

            "status":
                "available",

            "fallback_used":
                False,

            "reason":
                None,

            "activity_id":
                activity_id,

            "temperature_c":
                thermal_intelligence[
                    "temperature"
                ],

            "humidity_percent":
                thermal_intelligence[
                    "humidity"
                ],

            "heat_index_c":
                thermal_intelligence[
                    "heat_index"
                ],

            "apparent_temperature_c":
                thermal_intelligence[
                    "apparent_temperature"
                ],

            "wet_bulb_temperature_c":
                thermal_intelligence[
                    "wet_bulb_temperature"
                ],
        },

        # ====================================================
        # CONSUMPTION
        # ====================================================

        "consumption": {

            "current_kwh":
                current_consumption,

            "historical_baseline_kwh":
                avg_past_consumption,
        },

        # ====================================================
        # MODEL
        # ====================================================

        "model": {

            "prediction":
                prediction[
                    "prediction"
                ],

            "prediction_status":
                prediction[
                    "prediction_status"
                ],

            "label":
                prediction[
                    "label"
                ],

            "risk_level":
                prediction[
                    "risk_level"
                ],

            "risk_score":
                prediction[
                    "risk_score"
                ],

            "action":
                prediction[
                    "action"
                ],

            "action_code":
                prediction[
                    "action_code"
                ],

            "abnormal_probability":
                prediction[
                    "abnormal_probability"
                ],

            "consumption_ratio":
                prediction[
                    "consumption_ratio"
                ],

            "features":
                model_input.to_dict(
                    orient="records"
                )[0],
        },
    }


# ============================================================
# 16. LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PowerPulse Inference Test")
    print("=" * 70)

    try:

        model = load_model()

        print(
            "\nModel loaded successfully."
        )

        print(
            f"Model path: "
            f"{_MODEL_PATH_USED}"
        )

        test_input = create_model_input(
            current_consumption=100.0,
            avg_past_consumption=500.0,
            temperature=33.0,
            humidity=75.0,
            wind_speed=20.3,
        )

        print(
            "\nTest model input:"
        )

        print(
            test_input
        )

        result = predict_anomaly(
            model=model,
            model_input=test_input,
            current_consumption=100.0,
            avg_past_consumption=500.0,
        )

        print(
            "\nPrediction:"
        )

        print(
            result
        )

    except Exception as error:

        print(
            "\nERROR:"
        )

        print(
            error
        )
