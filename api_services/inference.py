from datetime import datetime
import os
import time
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

# IMPORTANT:
# Always load .env from the same directory as this inference file.
ENV_FILE = BASE_DIR / ".env"

load_dotenv(
    dotenv_path=ENV_FILE,
    override=True,
)

MODEL_FILENAME = "power_anomaly_model.pkl"
MODEL_PATH = BASE_DIR / MODEL_FILENAME

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

FORTYGUARD_URL = os.getenv(
    "FORTYGUARD_BASE_URL",
    "https://api.fortyguard.com/v1",
).strip().rstrip("/")


# ============================================================
# 2. MODEL FEATURE ORDER
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
# 5. MODEL
# ============================================================

def load_model():
    if not MODEL_PATH.exists():
        raise IntegrationError(
            f"Model file not found: {MODEL_PATH}"
        )

    try:
        return joblib.load(MODEL_PATH)
    except Exception as error:
        raise IntegrationError(
            f"Failed to load model '{MODEL_FILENAME}': {error}"
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
            f"Expected it in: {ENV_FILE}"
        )

    return api_key


def get_fortyguard_headers() -> Dict[str, str]:
    """
    FortyGuard documentation specifies `api-key`.
    Do not replace this with Bearer authentication.
    """

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

    if state not in STATE_COORDINATES:
        raise IntegrationError(
            f"Invalid US state: {state}"
        )

    return STATE_COORDINATES[state]


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

    temperature = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    wind_speed = current.get("wind_speed_10m")
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
        "temperature": float(temperature),
        "humidity": float(humidity),
        "wind_speed": float(wind_speed),
        "timestamp": str(timestamp),
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
        clean_timestamp = str(timestamp).replace(
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
        "[FortyGuard] submitting environment analysis..."
    )
    print(
        f"[FortyGuard] URL: {url}"
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

    if not isinstance(response_data, dict):
        raise IntegrationError(
            "FortyGuard returned an unexpected response: "
            f"{response.text}"
        )

    if response_data.get("error") is True:
        raise IntegrationError(
            "FortyGuard API rejected the request: "
            f"{response_data.get('message', response_data)}"
        )

    response_data_inner = response_data.get("data")

    if not isinstance(response_data_inner, dict):
        response_data_inner = {}

    activity_id = (
        response_data_inner.get("activity_id")
        or response_data.get("activity_id")
    )

    if not activity_id:
        raise IntegrationError(
            "FortyGuard did not return activity_id. "
            f"Response: {response_data}"
        )

    activity_id = str(activity_id)

    print(
        "[FortyGuard] activity submitted successfully:"
        f" {activity_id}"
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

    activity_id = str(activity_id).strip()

    if not activity_id:
        raise IntegrationError(
            "FortyGuard activity_id is empty."
        )

    url = (
        f"{FORTYGUARD_URL}"
        f"/status/{activity_id}"
    )

    print(
        "[FortyGuard] polling activity:"
        f" {activity_id}"
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
                f"FortyGuard status connection error: {error}"
            ) from error

        try:
            response_data = response.json()
        except ValueError as error:
            raise IntegrationError(
                "FortyGuard status returned invalid JSON. "
                f"HTTP {response.status_code}: "
                f"{response.text}"
            ) from error

        if response.status_code >= 400:
            raise IntegrationError(
                "FortyGuard status request failed "
                f"(HTTP {response.status_code}): "
                f"{response_data}"
            )

        if not isinstance(response_data, dict):
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
            f"[FortyGuard] "
            f"attempt {attempt}/{max_attempts} "
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

            result = data.get("result")

            if not isinstance(result, dict):
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
                or "FortyGuard activity failed."
            )

            raise IntegrationError(
                f"FortyGuard processing error: "
                f"{error_message}"
            )

        # ----------------------------------------------------
        # PROCESSING
        # ----------------------------------------------------

        if attempt < max_attempts:
            time.sleep(wait_seconds)

    raise TimeoutError(
        "FortyGuard activity did not complete within "
        f"{max_attempts * wait_seconds} seconds. "
        f"activity_id={activity_id}"
    )


# ============================================================
# 11. EXTRACT FORTYGUARD INTELLIGENCE
# ============================================================

def extract_fortyguard_intelligence(
    data: Dict[str, Any],
) -> Dict[str, Any]:

    if not isinstance(data, dict):
        raise IntegrationError(
            "FortyGuard response is not a dictionary."
        )

    result = data.get("result")

    if not isinstance(result, dict):
        result = data

    locations = result.get("locations")

    if (
        not isinstance(locations, list)
        or not locations
    ):

        single_location = result.get("location")

        if isinstance(single_location, dict):
            locations = [single_location]
        else:
            raise IntegrationError(
                "FortyGuard returned no valid locations."
            )

    location = locations[0]

    if not isinstance(location, dict):
        raise IntegrationError(
            "FortyGuard location structure is invalid."
        )

    parameters = location.get("parameters")

    if not isinstance(parameters, dict):
        parameters = {}

    def first_value(
        obj: Dict[str, Any],
        keys: list[str],
    ):

        for key in keys:

            if key not in obj:
                continue

            value = obj.get(key)

            if value is None:
                continue

            if isinstance(value, list):

                if not value:
                    continue

                value = value[0]

            return value

        return None

    temperature = first_value(
        location,
        [
            "temperature",
            "temp",
            "temperature_celsius",
        ],
    )

    if temperature is None:
        temperature = first_value(
            parameters,
            [
                "temperature",
                "temp",
                "temperature_celsius",
            ],
        )

    latitude = first_value(
        location,
        [
            "lat",
            "latitude",
        ],
    )

    longitude = first_value(
        location,
        [
            "lon",
            "longitude",
        ],
    )

    humidity = first_value(
        parameters,
        [
            "relative_humidity_percent",
            "relative_humidity",
            "humidity",
        ],
    )

    heat_index = first_value(
        parameters,
        [
            "heat_index_celsius",
            "heat_index",
        ],
    )

    apparent_temperature = first_value(
        parameters,
        [
            "apparent_temperature_celsius",
            "apparent_temperature",
            "feels_like",
        ],
    )

    wet_bulb_temperature = first_value(
        parameters,
        [
            "wet_bulb_temperature_celsius",
            "wet_bulb_temperature",
            "wet_bulb",
        ],
    )

    return {
        "temperature": (
            float(temperature)
            if temperature is not None
            else None
        ),

        "latitude": (
            float(latitude)
            if latitude is not None
            else None
        ),

        "longitude": (
            float(longitude)
            if longitude is not None
            else None
        ),

        "humidity": (
            float(humidity)
            if humidity is not None
            else None
        ),

        "heat_index": (
            float(heat_index)
            if heat_index is not None
            else None
        ),

        "apparent_temperature": (
            float(apparent_temperature)
            if apparent_temperature is not None
            else None
        ),

        "wet_bulb_temperature": (
            float(wet_bulb_temperature)
            if wet_bulb_temperature is not None
            else None
        ),
    }


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

    current = float(current_consumption)
    baseline = float(avg_past_consumption)

    temp = float(temperature)
    hum = float(humidity)
    wind = float(wind_speed)

    epsilon = 1e-6

    difference = current - baseline

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
        "Electricity_Consumed": [current],
        "Temperature": [temp],
        "Humidity": [hum],
        "Wind_Speed": [wind],
        "Avg_Past_Consumption": [baseline],
        "Difference": [difference],
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
# 13. MODEL PREDICTION
# ============================================================

def predict_anomaly(
    model,
    model_input: pd.DataFrame,
    current_consumption: float,
    avg_past_consumption: float,
) -> Dict[str, Any]:

    current = float(current_consumption)
    baseline = float(avg_past_consumption)

    epsilon = 1e-6

    ratio = (
        current /
        (baseline + epsilon)
    )

    # --------------------------------------------------------
    # Normal range
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Critical under-consumption
    # --------------------------------------------------------

    if current <= 0.0 or ratio < 0.25:

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

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    try:
        model_prediction = int(
            model.predict(model_input)[0]
        )
    except Exception as error:
        raise IntegrationError(
            f"Model prediction failed: {error}"
        ) from error

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

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

                abnormal_index = classes.index(1)

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

    if probability <= 0.50:

        final_prediction = 0
        prediction_status = "Normal"
        label = "Normal"

    else:

        final_prediction = 1
        prediction_status = "Potentially Abnormal"
        label = "Abnormal"

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Normal prediction overrides risk
    # --------------------------------------------------------

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
# 14. COMPLETE POWERGUARD ANALYSIS
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

    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    latitude, longitude = (
        get_state_coordinates(state)
    )

    # --------------------------------------------------------
    # WEATHER
    # --------------------------------------------------------

    weather = get_current_weather(
        latitude,
        longitude,
    )

    # --------------------------------------------------------
    # FORTYGUARD
    #
    # IMPORTANT:
    # FortyGuard is REQUIRED.
    # There is NO fallback/mock here.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # VALIDATE REAL FORTYGUARD DATA
    # --------------------------------------------------------

    fg_temperature = (
        thermal_intelligence["temperature"]
    )

    fg_humidity = (
        thermal_intelligence["humidity"]
    )

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

    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    model_input = create_model_input(
        current_consumption=current_consumption,
        avg_past_consumption=avg_past_consumption,
        temperature=fg_temperature,
        humidity=fg_humidity,
        wind_speed=weather["wind_speed"],
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = load_model()

    prediction = predict_anomaly(
        model=model,
        model_input=model_input,
        current_consumption=current_consumption,
        avg_past_consumption=avg_past_consumption,
    )

    # --------------------------------------------------------
    # FINAL RESPONSE
    #
    # The UI reads this object directly.
    # --------------------------------------------------------

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
            "timestamp": weather["timestamp"],
        },

        "fortyguard": {

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
                model_input.to_dict(
                    orient="records"
                )[0]
            ),
        },
    }