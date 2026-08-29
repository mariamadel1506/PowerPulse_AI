import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# ============================================================
# 1. إعدادات الملفات
# ============================================================

DATA_PATH = (
    "C:/Users/hp/Desktop/Hackathon power pluse/data/smart_meter_data.csv"
)

MODEL_OUTPUT_PATH = "power_anomaly_model.pkl"


# ============================================================
# 2. قراءة البيانات
# ============================================================

df = pd.read_csv(DATA_PATH)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)


# ============================================================
# 3. تحويل البيانات من Dataset Scale إلى Real Scale
#
# مهم جدًا:
# يجب أن تكون نفس التحويلات المستخدمة في inference.py
# ============================================================

df["Electricity_Consumed"] = (
    df["Electricity_Consumed"] * 1000.0
)

df["Avg_Past_Consumption"] = (
    df["Avg_Past_Consumption"] * 1000.0
)

df["Temperature"] = (
    df["Temperature"] * 60.0 - 10.0
)

df["Humidity"] = (
    df["Humidity"] * 100.0
)

df["Wind_Speed"] = (
    df["Wind_Speed"] * 100.0
)


# ============================================================
# 4. Feature Engineering
# ============================================================

df["Difference"] = (
    df["Electricity_Consumed"]
    - df["Avg_Past_Consumption"]
)

df["Consumption_Ratio"] = (
    df["Electricity_Consumed"]
    / (df["Avg_Past_Consumption"] + 1e-6)
)

df["Consumption_Change_Percentage"] = (
    df["Difference"]
    / (df["Avg_Past_Consumption"] + 1e-6)
) * 100.0

df["Temp_Consumption_Interaction"] = (
    df["Temperature"]
    * df["Electricity_Consumed"]
)

df["Heatwave_Anomaly_Risk"] = (
    (
        (df["Temperature"] > 35.0)
        &
        (df["Consumption_Ratio"] > 1.10)
    )
).astype(int)


# ============================================================
# 5. تنظيف الـOriginal Labels
# ============================================================

df["Anomaly_Label"] = (
    df["Anomaly_Label"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# 6. Business Re-labeling
#
# الحالات التالية تعتبر Abnormal:
#
# Ratio < 0.60
# Ratio > 1.70
# أو الـOriginal Dataset قال Abnormal
# ============================================================

is_theft_drop = (
    df["Consumption_Ratio"] < 0.60
)

is_severe_spike = (
    df["Consumption_Ratio"] > 1.70
)

is_original_abnormal = (
    df["Anomaly_Label"] == "abnormal"
)

df["Final_Target"] = (
    is_theft_drop
    | is_severe_spike
    | is_original_abnormal
).astype(int)


# ============================================================
# 7. تنظيف البيانات
# ============================================================

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

df.dropna(
    inplace=True
)

df.reset_index(
    drop=True,
    inplace=True
)


# ============================================================
# 8. Features
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


X = df[FEATURE_ORDER].copy()

y = df["Final_Target"].copy()


# ============================================================
# 9. Train / Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# ============================================================
# 10. Random Forest
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)


rf_model.fit(
    X_train,
    y_train
)


# ============================================================
# 11. Evaluation
# ============================================================

y_pred = rf_model.predict(X_test)

print("\n==========================================")
print("MODEL EVALUATION")
print("==========================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Abnormal"
        ],
        zero_division=0,
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 12. Save Model
# ============================================================

joblib.dump(
    rf_model,
    "power_anomaly_model.pkl"
)

print(
    f"\nModel saved successfully: power_anomaly_model.pkl"
)


# ============================================================
# 13. Instant Verification
# ============================================================

loaded_model = joblib.load(
    "power _anomaly_model.pkl"
)


test_current = 100.0
test_baseline = 500.0
test_temperature = 33.0
test_humidity = 75.0
test_wind = 20.3


test_difference = (
    test_current
    - test_baseline
)

test_ratio = (
    test_current
    / (test_baseline + 1e-6)
)

test_change_pct = (
    test_difference
    / (test_baseline + 1e-6)
) * 100.0

test_interaction = (
    test_temperature
    * test_current
)

test_heatwave = int(
    test_temperature > 35.0
    and test_ratio > 1.10
)


test_sample = pd.DataFrame(
    [{
        "Electricity_Consumed": test_current,
        "Temperature": test_temperature,
        "Humidity": test_humidity,
        "Wind_Speed": test_wind,
        "Avg_Past_Consumption": test_baseline,
        "Difference": test_difference,
        "Consumption_Ratio": test_ratio,
        "Consumption_Change_Percentage": test_change_pct,
        "Temp_Consumption_Interaction": test_interaction,
        "Heatwave_Anomaly_Risk": test_heatwave,
    }],
    columns=FEATURE_ORDER,
)


test_prediction = int(
    loaded_model.predict(test_sample)[0]
)

test_probability = float(
    loaded_model.predict_proba(test_sample)[0][1]
)


print("\n==========================================")
print("INSTANT VERIFICATION")
print("==========================================")

print(
    f"Ratio: {test_ratio:.4f}"
)

print(
    f"Prediction: "
    f"{'Abnormal' if test_prediction == 1 else 'Normal'}"
)

print(
    "Probability [Normal, Abnormal]: "
    f"[{1 - test_probability:.2f}, "
    f"{test_probability:.2f}]"
)
