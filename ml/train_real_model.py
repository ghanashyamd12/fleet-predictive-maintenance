import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =========================
# 1. Load NASA turbofan dataset
# =========================

data_path = "data/CMAPSSData/train_FD001.txt"


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

# =========================
# 3. Prepare features
# =========================

X = df[[f"sensor_{i}" for i in range(1, 22)]]
y = df["RUL"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. Train model
# =========================

model = RandomForestRegressor(
    n_estimators=80,      # smaller → smaller file size
    max_depth=10,
    random_state=42,
    n_jobs=-1
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
# 6. Save model (SMALL SIZE)
# =========================

os.makedirs("ml/model", exist_ok=True)

joblib.dump(model, "ml/model/rul_model.pkl", compress=3)

print("✅ Real model saved at ml/model/rul_model.pkl")
