import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# =========================
# 1. Load NASA turbofan dataset
# =========================

data_path = "ml/data/CMAPSSData/train_FD001.txt"

cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

df = pd.read_csv(data_path, sep=" ", header=None)
df = df.iloc[:, :26]
df.columns = cols

# =========================
# 2. Create Remaining Useful Life (RUL)
# =========================

max_cycles = df.groupby("unit")["cycle"].max().reset_index()
max_cycles.columns = ["unit", "max_cycle"]

df = df.merge(max_cycles, on="unit")
df["RUL"] = df["max_cycle"] - df["cycle"]

# ---- NASA standard: clip RUL ----
RUL_CLIP = 130
df["RUL"] = df["RUL"].clip(upper=RUL_CLIP)

# =========================
# 3. Prepare features
# =========================

X = df[[f"sensor_{i}" for i in range(1, 22)]]
y = df["RUL"]

# ---- Feature scaling ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# 4. Train FAST + VALID model
# =========================

model = RandomForestRegressor(
    n_estimators=40,   # balanced accuracy + speed
    max_depth=8,
    random_state=42,
    n_jobs=1
)

model.fit(X_train, y_train)

# =========================
# 5. Evaluate
# =========================

preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)

print("RMSE:", round(rmse, 2))
print("MAE :", round(mae, 2))

# =========================
# 6. Save model + scaler
# =========================

os.makedirs("ml/model", exist_ok=True)

joblib.dump(
    {"model": model, "scaler": scaler},
    "ml/model/rul_model.pkl",
    compress=3,
)

print("✅ FINAL industry model saved.")
