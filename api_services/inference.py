from datetime import datetime
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# 1. Environment
# ============================================================

load_dotenv()


# inference.py موجود غالبًا داخل:
# project/
# ├── main.py
# ├── generate_html.py
# ├── power_anomaly_model.pkl
# └── api_services/
#     └── inference.py
#
# لذلك نستخدم parent.parent للوصول إلى root المشروع.

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


FORTYGUARD_BASE_URL = os.getenv(
    "FORTYGUARD_BASE_URL",
    "https://api.fortyguard.com/v1",
).rstrip("/")


# ============================================================
# 2. Model Configuration
# ============================================================

# ندعم الاسمين لأن عندك نسخة محفوظة بالمسافة ونسخة بدونها.
MODEL_CANDIDATES = [
    PROJECT_ROOT / "power_anomaly_model.pkl",
    PROJECT_ROOT / "power _anomaly_model.pkl",
    CURRENT_DIR / "power_anomaly_model.pkl",
    CURRENT_DIR / "power _anomaly_model.pkl",
]


# نفس الترتيب المستخدم أثناء التدريب بالضبط.
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
# 3. US State Coordinates
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
# 4. Custom Error
# ============================================================

class IntegrationError(Exception):
    pass


# ============================================================
# 5. Load Model
# ============================================================

def load_model():
    """
    Loads the trained RandomForest model.

    Supports both:
        power_anomaly_model.pkl
        power _anomaly_model.pkl
    """

    existing_paths = [
        path
        for path in MODEL_CANDIDATES
        if path.exists()
    ]

    if not existing_paths:
        checked = "\n".join(
            str(path)
            for path in MODEL_CANDIDATES
        )

        raise IntegrationError(
            "Trained model file was not found.\n"
            "Checked these locations:\n"
            f"{checked}"
        )

    # أول ملف موجود
    model_path = existing_paths[0]

    try:
        model = joblib.load(model_path)
    except Exception as error:
        raise IntegrationError(
            f"Failed to load trained model "
            f"'{model_path}': {error}"
        ) from error

    # تأكيد أن الملف فعلاً model قابل للتنبؤ
    if not hasattr(model, "predict"):
        raise IntegrationError(
            f"Loaded object from '{model_path}' "
            "does not provide a predict() method."
        )

    return model


# ============================================================
# 6. FortyGuard API Key
# ============================================================

def get_fortyguard_api_key() -> str:

    api_key = os.getenv("FORTYGUARD_API_KEY")

    if not api_key:
        raise IntegrationError(
            "FORTYGUARD_API_KEY is missing. "
            "Add it to your .env file or deployment environment variables."
        )

    api_key = api_key.strip()

    if not api_key:
        raise IntegrationError(
            "FORTYGUARD_API_KEY is empty."
        )

    return api_key


# ============================================================
# 7. State Coordinates
# ============================================================

def get_state_coordinates(
    state: str,
) -> tuple[float, float]:

    normalized_state = state.strip()

    if normalized_state not in STATE_COORDINATES:
        raise IntegrationError(
            f"Invalid US state: {state}"
        )

    return STATE_COORDINATES[normalized_state]


# ============================================================
# 8. Open-Meteo Weather
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

        response.raise_for_status()

    except requests.Timeout as error:
        raise IntegrationError(
            f"Weather API timeout: {error}"
        ) from error

    except requests.RequestException as error:
        raise IntegrationError(
            f"Weather API connection error: {error}"
        ) from error

    try:
        data = response.json()
    except ValueError as error:
        raise IntegrationError(
            "Weather API returned invalid JSON."
        ) from error

    current = data.get("current")

    if not isinstance(current, dict):
        raise IntegrationError(
            "Weather API returned no current weather data."
        )

    temperature = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    wind_speed = current.get("wind_speed_10m")
    timestamp = current.get("time")

    if any(
        value is None
        for value in [
            temperature,
            humidity,
            wind_speed,
            timestamp,
        ]
    ):
        raise IntegrationError(
            "Weather API returned incomplete current data."
        )

    return {
        "temperature": float(temperature),
        "humidity": float(humidity),
        "wind_speed": float(wind_speed),
        "timestamp": str(timestamp),
    }


# ============================================================
# 9. Timestamp Parser
# ============================================================

def parse_weather_timestamp(timestamp: str) -> datetime:
    """
    Open-Meteo may return:
        2026-08-29T16:00

    or an ISO timestamp containing an offset.

    FortyGuard needs:
        start_date
        start_time
    """

    if not timestamp:
        raise IntegrationError(
            "Weather timestamp is empty."
        )

    clean_timestamp = str(timestamp).strip()

    try:
        return datetime.fromisoformat(
            clean_timestamp
        )

    except ValueError:
        pass

    # Extra defensive handling for trailing Z
    try:
        return datetime.fromisoformat(
            clean_timestamp.replace(
                "Z",
                "+00:00",
            )
        )

    except ValueError as error:
        raise IntegrationError(
            f"Invalid weather timestamp: {timestamp}"
        ) from error


# ============================================================
# 10. FortyGuard Headers
# ============================================================

def fortyguard_headers() -> Dict[str, str]:

    api_key = get_fortyguard_api_key()

    return {
        # This is exactly the header shown
        # in the FortyGuard documentation.
        "api-key": api_key,

        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# ============================================================
# 11. FortyGuard Submit
# ============================================================

def submit_fortyguard(
    latitude: float,
    longitude: float,
    temperature: float,
    timestamp: str,
) -> str:

    timestamp_dt = parse_weather_timestamp(
        timestamp
    )

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

    url = (
        f"{FORTYGUARD_BASE_URL}"
        "/env_params"
    )

    try:
        response = requests.post(
            url,
            headers=fortyguard_headers(),
            json=payload,
            timeout=45,
        )

    except requests.Timeout as error:
        raise IntegrationError(
            f"FortyGuard submit timeout: {error}"
        ) from error

    except requests.RequestException as error:
        raise IntegrationError(
            f"FortyGuard connection error: {error}"
        ) from error

    # Parse JSON even for error responses,
    # because FortyGuard may provide useful details.
    try:
        response_data = response.json()
    except ValueError:
        response_data = None

    if response.status_code == 401:
        raise IntegrationError(
            "FortyGuard rejected the API key (401 Unauthorized)."
        )

    if response.status_code == 403:
        raise IntegrationError(
            "FortyGuard denied access (403 Forbidden)."
        )

    if response.status_code == 429:
        raise IntegrationError(
            "FortyGuard rate limit exceeded (429)."
        )

    if response.status_code >= 400:

        detail = (
            response_data
            if response_data is not None
            else response.text
        )

        raise IntegrationError(
            "FortyGuard submission failed "
            f"(HTTP {response.status_code}): {detail}"
        )

    if not isinstance(response_data, dict):
        raise IntegrationError(
            "FortyGuard submission returned "
            "a non-JSON response."
        )

    # Expected:
    #
    # {
    #   "error": false,
    #   "status_code": 200,
    #   "message": "...",
    #   "data": {
    #       "activity_id": "..."
    #   }
    # }

    if response_data.get("error") is True:
        raise IntegrationError(
            "FortyGuard reported an API error: "
            f"{response_data.get('message', response_data)}"
        )

    data = response_data.get("data")

    if not isinstance(data, dict):
        data = {}

    activity_id = (
        data.get("activity_id")
        or response_data.get("activity_id")
    )

    if not activity_id:
        raise IntegrationError(
            "FortyGuard submission succeeded but "
            "no activity_id was returned. "
            f"Response: {response_data}"
        )

    return str(activity_id)


# ============================================================
# 12. FortyGuard Status Polling
# ============================================================

def get_fortyguard_result(
    activity_id: str,
    max_attempts: int = 90,
    wait_seconds: int = 2,
) -> Dict[str, Any]:

    """
    Poll FortyGuard until the analysis is completed.

    Documented Processing response:

    {
        "error": false,
        "status_code": 200,
        "message": "Processing",
        "data": {
            "activity_id": "...",
            "status": "Processing"
        }
    }

    Completed response:

    {
        "error": false,
        "status_code": 200,
        "message": "Completed",
        "data": {
            "activity_id": "...",
            "status": "Completed",
            "result": {...}
        }
    }
    """

    if not activity_id:
        raise IntegrationError(
            "FortyGuard activity_id is empty."
        )

    url = (
        f"{FORTYGUARD_BASE_URL}"
        f"/status/{activity_id}"
    )

    headers = fortyguard_headers()

    last_status = None

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=45,
            )

        except requests.Timeout as error:
            raise IntegrationError(
                f"FortyGuard status timeout: {error}"
            ) from error

        except requests.RequestException as error:
            raise IntegrationError(
                f"FortyGuard status connection error: {error}"
            ) from error

        try:
            response_data = response.json()
        except ValueError as error:
            raise IntegrationError(
                "FortyGuard status endpoint returned "
                f"invalid JSON. HTTP {response.status_code}"
            ) from error

        if response.status_code == 401:
            raise IntegrationError(
                "FortyGuard rejected the API key "
                "while checking status."
            )

        if response.status_code == 403:
            raise IntegrationError(
                "FortyGuard denied status access."
            )

        if response.status_code == 404:
            raise IntegrationError(
                "FortyGuard activity was not found: "
                f"{activity_id}"
            )

        if response.status_code == 429:
            raise IntegrationError(
                "FortyGuard rate limit exceeded "
                "while polling status."
            )

        if response.status_code >= 500:
            raise IntegrationError(
                "FortyGuard server error while "
                f"polling status: HTTP {response.status_code}"
            )

        if response.status_code >= 400:
            raise IntegrationError(
                "FortyGuard status request failed: "
                f"HTTP {response.status_code} - "
                f"{response_data}"
            )

        if response_data.get("error") is True:
            raise IntegrationError(
                "FortyGuard returned an error while "
                f"checking status: {response_data}"
            )

        data = response_data.get("data")

        if not isinstance(data, dict):
            data = {}

        status = (
            data.get("status")
            or response_data.get("status")
            or response_data.get("message")
            or ""
        )

        status = str(status).strip().lower()

        last_status = status

        # ----------------------------------------------------
        # COMPLETED
        # ----------------------------------------------------

        if status in {
            "completed",
            "complete",
            "success",
            "successful",
            "done",
            "ok",
        }:

            result = data.get("result")

            if not isinstance(result, dict):
                raise IntegrationError(
                    "FortyGuard reported Completed but "
                    "did not return a valid result object."
                )

            # Keep the exact structure expected by the
            # extraction function.
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
                or "Unknown FortyGuard processing error"
            )

            raise IntegrationError(
                "FortyGuard processing failed: "
                f"{message}"
            )

        # ----------------------------------------------------
        # PROCESSING
        # ----------------------------------------------------

        if attempt < max_attempts:
            time.sleep(wait_seconds)

    raise TimeoutError(
        "FortyGuard analysis did not complete "
        f"within the polling window. "
        f"activity_id={activity_id}, "
        f"last_status={last_status}, "
        f"attempts={max_attempts}"
    )


# ============================================================
# 13. Safe Value Extraction
# ============================================================

def _extract_first_value(
    obj: Dict[str, Any],
    keys: list[str],
) -> Optional[Any]:

    if not isinstance(obj, dict):
        return None

    for key in keys:

        if key not in obj:
            continue

        value = obj.get(key)

        if value is None:
            continue

        # FortyGuard parameters can be arrays.
        if isinstance(value, list):

            if not value:
                continue

            value = value[0]

        return value

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
# 14. Extract FortyGuard Intelligence
# ============================================================

def extract_fortyguard_intelligence(
    data: Dict[str, Any],
) -> Dict[str, Any]:

    if not isinstance(data, dict):
        raise IntegrationError(
            "FortyGuard response is not a dictionary."
        )

    # get_fortyguard_result returns data directly:
    #
    # {
    #     "activity_id": "...",
    #     "status": "Completed",
    #     "result": {...}
    # }

    result = data.get("result")

    if not isinstance(result, dict):
        raise IntegrationError(
            "FortyGuard completed response contains "
            "no valid result object."
        )

    locations = result.get("locations")

    if not isinstance(
        locations,
        list,
    ) or not locations:

        # Defensive support for a singular location
        single_location = result.get("location")

        if isinstance(
            single_location,
            dict,
        ):
            locations = [
                single_location
            ]

        else:
            raise IntegrationError(
                "FortyGuard result contains no locations."
            )

    location = locations[0]

    if not isinstance(location, dict):
        raise IntegrationError(
            "FortyGuard location object is invalid."
        )

    parameters = location.get(
        "parameters",
        {},
    )

    if not isinstance(parameters, dict):
        parameters = {}

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    latitude = _safe_float(
        _extract_first_value(
            location,
            [
                "lat",
                "latitude",
            ],
        )
    )

    longitude = _safe_float(
        _extract_first_value(
            location,
            [
                "lon",
                "longitude",
            ],
        )
    )

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    temperature = _safe_float(
        _extract_first_value(
            location,
            [
                "temperature",
                "temp",
                "temperature_celsius",
            ],
        )
    )

    # --------------------------------------------------------
    # FortyGuard parameters
    # --------------------------------------------------------

    humidity = _safe_float(
        _extract_first_value(
            parameters,
            [
                "relative_humidity_percent",
                "relative_humidity",
                "humidity",
            ],
        )
    )

    heat_index = _safe_float(
        _extract_first_value(
            parameters,
            [
                "heat_index_celsius",
                "heat_index",
            ],
        )
    )

    apparent_temperature = _safe_float(
        _extract_first_value(
            parameters,
            [
                "apparent_temperature_celsius",
                "apparent_temperature",
                "feels_like",
            ],
        )
    )

    wet_bulb_temperature = _safe_float(
        _extract_first_value(
            parameters,
            [
                "wet_bulb_temperature_celsius",
                "wet_bulb_temperature",
                "wet_bulb",
            ],
        )
    )

    # --------------------------------------------------------
    # Additional FortyGuard data
    # --------------------------------------------------------

    precipitation = _safe_float(
        _extract_first_value(
            parameters,
            [
                "precipitation_mm",
                "precipitation",
            ],
        )
    )

    cloud_cover = _safe_float(
        _extract_first_value(
            parameters,
            [
                "cloud_cover_octas",
                "cloud_cover",
            ],
        )
    )

    air_quality = _safe_float(
        _extract_first_value(
            parameters,
            [
                "air_quality:idx",
            ],
        )
    )

    pm25 = _safe_float(
        _extract_first_value(
            parameters,
            [
                "air_quality_pm2p5:idx",
            ],
        )
    )

    pm10 = _safe_float(
        _extract_first_value(
            parameters,
            [
                "air_quality_pm10:idx",
            ],
        )
    )

    co2 = _safe_float(
        _extract_first_value(
            parameters,
            [
                "co2_ppm",
            ],
        )
    )

    methane = _safe_float(
        _extract_first_value(
            parameters,
            [
                "methane_ppb",
            ],
        )
    )

    # --------------------------------------------------------
    # Solar Irradiance
    # --------------------------------------------------------

    solar = result.get(
        "locations",
        [{}],
    )[0].get(
        "solar_irradiance",
        {},
    )

    if not isinstance(solar, dict):
        solar = {}

    clear_sky = solar.get(
        "clear_sky",
        {},
    )

    if not isinstance(
        clear_sky,
        dict,
    ):
        clear_sky = {}

    ghi = _safe_float(
        clear_sky.get("ghi")
    )

    dni = _safe_float(
        clear_sky.get("dni")
    )

    dhi = _safe_float(
        clear_sky.get("dhi")
    )

    solar_description = solar.get(
        "description"
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "temperature": temperature,

        "latitude": latitude,
        "longitude": longitude,

        "humidity": humidity,

        "heat_index": heat_index,

        "apparent_temperature": (
            apparent_temperature
        ),

        "wet_bulb_temperature": (
            wet_bulb_temperature
        ),

        "precipitation": precipitation,

        "cloud_cover": cloud_cover,

        "air_quality": air_quality,

        "pm2_5": pm25,

        "pm10": pm10,

        "co2": co2,

        "methane": methane,

        "solar_irradiance": {
            "ghi": ghi,
            "dni": dni,
            "dhi": dhi,
            "description": solar_description,
        },
    }


# ============================================================
# 15. Feature Engineering
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

    if baseline <= 0:
        raise IntegrationError(
            "Historical baseline consumption "
            "must be greater than zero."
        )

    epsilon = 1e-6

    difference = (
        current - baseline
    )

    ratio = (
        current
        / (baseline + epsilon)
    )

    change_percentage = (
        difference
        / (baseline + epsilon)
    ) * 100.0

    interaction = (
        temp * current
    )

    heatwave = int(
        temp > 35.0
        and ratio > 1.10
    )

    features = {
        "Electricity_Consumed": current,
        "Temperature": temp,
        "Humidity": hum,
        "Wind_Speed": wind,
        "Avg_Past_Consumption": baseline,
        "Difference": difference,
        "Consumption_Ratio": ratio,
        "Consumption_Change_Percentage": (
            change_percentage
        ),
        "Temp_Consumption_Interaction": (
            interaction
        ),
        "Heatwave_Anomaly_Risk": heatwave,
    }

    return pd.DataFrame(
        [features],
        columns=FEATURE_ORDER,
    )


# ============================================================
# 16. Model Prediction
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

    ratio = (
        current
        / (baseline + 1e-6)
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:
        prediction = int(
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

    probability = None

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
                f"Model probability prediction failed: "
                f"{error}"
            ) from error

    else:

        probability = (
            1.0
            if prediction == 1
            else 0.0
        )

    probability = max(
        0.0,
        min(
            1.0,
            probability,
        ),
    )

    # --------------------------------------------------------
    # Final classification
    # --------------------------------------------------------

    label = (
        "Abnormal"
        if prediction == 1
        else "Normal"
    )

    prediction_status = label

    # --------------------------------------------------------
    # Risk calculation
    #
    # Keep the model prediction independent from
    # FortyGuard. FortyGuard provides environmental
    # intelligence; the ML model remains responsible
    # for the anomaly prediction.
    # --------------------------------------------------------

    if (
        current <= 0.0
        or ratio < 0.60
        or ratio > 1.70
        or prediction == 1
    ):

        if (
            current <= 0.0
            or ratio < 0.25
            or ratio > 4.0
        ):

            risk_level = "CRITICAL"
            risk_score = 100.0
            action_code = "CRITICAL_INSPECT"

            action = (
                "Urgent field inspection is required "
                "to investigate the severe consumption "
                "deviation and verify meter and grid conditions."
            )

        elif (
            ratio < 0.50
            or ratio > 2.00
        ):

            risk_level = "HIGH"
            risk_score = 75.0
            action_code = "INSPECT"

            action = (
                "Dispatch a field audit team to perform "
                "physical meter examination and diagnostic assessment."
            )

        else:

            risk_level = "MEDIUM"
            risk_score = 50.0
            action_code = "INVESTIGATE"

            action = (
                "Conduct remote telemetry analysis and "
                "investigate the consumption deviation "
                "against localized environmental conditions."
            )

    else:

        risk_level = "LOW"
        risk_score = 0.0
        action_code = "MONITOR"

        action = (
            "Maintain routine operational monitoring."
        )

    return {
        "prediction": prediction,
        "prediction_status": prediction_status,
        "label": label,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "action": action,
        "action_code": action_code,
        "abnormal_probability": round(
            probability,
            4,
        ),
        "consumption_ratio": round(
            ratio,
            4,
        ),
    }


# ============================================================
# 17. Complete PowerPulse Analysis
# ============================================================

def run_powerguard_analysis(
    state: str,
    current_consumption: float,
    avg_past_consumption: float,
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if current_consumption < 0:
        raise IntegrationError(
            "Current consumption cannot be negative."
        )

    if avg_past_consumption <= 0:
        raise IntegrationError(
            "Historical baseline consumption "
            "must be greater than zero."
        )

    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    latitude, longitude = (
        get_state_coordinates(state)
    )

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    weather = get_current_weather(
        latitude=latitude,
        longitude=longitude,
    )

    # --------------------------------------------------------
    # FortyGuard
    #
    # IMPORTANT:
    # There is NO fallback/mock here.
    # The real API must respond successfully.
    # --------------------------------------------------------

    activity_id = submit_fortyguard(
        latitude=latitude,
        longitude=longitude,
        temperature=weather["temperature"],
        timestamp=weather["timestamp"],
    )

    fortyguard_data = get_fortyguard_result(
        activity_id=activity_id,
        max_attempts=90,
        wait_seconds=2,
    )

    thermal_intelligence = (
        extract_fortyguard_intelligence(
            fortyguard_data
        )
    )

    # --------------------------------------------------------
    # Use FortyGuard environmental values when available.
    #
    # If FortyGuard doesn't provide a specific parameter,
    # we use Open-Meteo ONLY for that missing raw input.
    #
    # This is not a fake FortyGuard result.
    # --------------------------------------------------------

    effective_temperature = (
        thermal_intelligence["temperature"]
        if thermal_intelligence["temperature"]
        is not None
        else weather["temperature"]
    )

    effective_humidity = (
        thermal_intelligence["humidity"]
        if thermal_intelligence["humidity"]
        is not None
        else weather["humidity"]
    )

    # --------------------------------------------------------
    # Model Input
    # --------------------------------------------------------

    model_input = create_model_input(
        current_consumption=current_consumption,
        avg_past_consumption=avg_past_consumption,
        temperature=effective_temperature,
        humidity=effective_humidity,
        wind_speed=weather["wind_speed"],
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = predict_anomaly(
        model=model,
        model_input=model_input,
        current_consumption=current_consumption,
        avg_past_consumption=avg_past_consumption,
    )

    # --------------------------------------------------------
    # FortyGuard result metadata
    # --------------------------------------------------------

    fg_status = str(
        fortyguard_data.get(
            "status",
            "Completed",
        )
    )

    # الـUI الحالي عندك بيفحص specifically:
    # fg.status === "available"
    #
    # لذلك نخلي status = available مع الاحتفاظ
    # بالـactual FortyGuard status في processing_status.

    fortyguard_result = {
        "status": "available",

        "processing_status": fg_status,

        "fallback_used": False,

        "reason": None,

        "activity_id": activity_id,

        "temperature_c": (
            thermal_intelligence["temperature"]
        ),

        "humidity_percent": (
            thermal_intelligence["humidity"]
        ),

        "heat_index_c": (
            thermal_intelligence["heat_index"]
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

        "precipitation_mm": (
            thermal_intelligence[
                "precipitation"
            ]
        ),

        "cloud_cover_octas": (
            thermal_intelligence[
                "cloud_cover"
            ]
        ),

        "air_quality_index": (
            thermal_intelligence[
                "air_quality"
            ]
        ),

        "pm2_5_index": (
            thermal_intelligence[
                "pm2_5"
            ]
        ),

        "pm10_index": (
            thermal_intelligence[
                "pm10"
            ]
        ),

        "co2_ppm": (
            thermal_intelligence[
                "co2"
            ]
        ),

        "methane_ppb": (
            thermal_intelligence[
                "methane"
            ]
        ),

        "solar_irradiance": (
            thermal_intelligence[
                "solar_irradiance"
            ]
        ),
    }

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {

        # ====================================================
        # Location
        # ====================================================

        "location": {
            "state": state,
            "latitude": latitude,
            "longitude": longitude,
        },

        # ====================================================
        # Weather
        # ====================================================

        "weather": {
            "temperature_c": (
                effective_temperature
            ),

            "humidity_percent": (
                effective_humidity
            ),

            "wind_speed_kmh": (
                weather["wind_speed"]
            ),

            "timestamp": (
                weather["timestamp"]
            ),
        },

        # ====================================================
        # FortyGuard
        # ====================================================

        "fortyguard": fortyguard_result,

        # ====================================================
        # Consumption
        # ====================================================

        "consumption": {

            "current_kwh": float(
                current_consumption
            ),

            "historical_baseline_kwh": float(
                avg_past_consumption
            ),
        },

        # ====================================================
        # Model
        # ====================================================

        "model": {

            "prediction": (
                prediction["prediction"]
            ),

            "prediction_status": (
                prediction[
                    "prediction_status"
                ]
            ),

            "label": (
                prediction["label"]
            ),

            "risk_level": (
                prediction["risk_level"]
            ),

            "risk_score": (
                prediction["risk_score"]
            ),

            "action": (
                prediction["action"]
            ),

            "action_code": (
                prediction["action_code"]
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
                model_input
                .to_dict(
                    orient="records"
                )[0]
            ),
        },

        # ====================================================
        # Decision Support
        #
        # Added so your UI can safely use:
        # data.decision_support.action_code
        # ====================================================

        "decision_support": {

            "action_code": (
                prediction["action_code"]
            ),

            "action": (
                prediction["action"]
            ),

            "risk_level": (
                prediction["risk_level"]
            ),

            "risk_score": (
                prediction["risk_score"]
            ),
        },
    }
