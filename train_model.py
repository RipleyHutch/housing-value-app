import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/Housing_Hamilton_Compressed.csv.gz"
ARTIFACTS_DIR = "artifacts"

FEATURES = [
    "CALC_ACRES",
    "LAND_USE_CODE_DESC",
    "NEIGHBORHOOD_CODE_DESC",
    "ZONING_DESC",
    "PROPERTY_TYPE_CODE_DESC"
]

TARGET = "APPRAISED_VALUE"

NUMERIC_FEATURES = ["CALC_ACRES"]

CATEGORICAL_FEATURES = [
    "LAND_USE_CODE_DESC",
    "NEIGHBORHOOD_CODE_DESC",
    "ZONING_DESC",
    "PROPERTY_TYPE_CODE_DESC"
]

np.random.seed(42)
tf.random.set_seed(42)

# Load data
df = pd.read_csv(DATA_PATH)

df = df[FEATURES + [TARGET]].copy()

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=FEATURES + [TARGET])
df = df[df[TARGET] > 0]

X = df[FEATURES]
y = df[TARGET]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
)

X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

if hasattr(X_train_p, "toarray"):
    X_train_p = X_train_p.toarray()
    X_test_p = X_test_p.toarray()

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_p.shape[1],)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1),
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mean_absolute_error"],
)

# Train
model.fit(
    X_train_p,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1,
)

# Evaluate
loss, mae = model.evaluate(X_test_p, y_test, verbose=0)

print("Test MSE:", loss)
print("Test MAE:", mae)

# Save
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

model.save("artifacts/housing_model.keras")

joblib.dump(preprocessor, "artifacts/preprocessor.pkl")
joblib.dump(FEATURES, "artifacts/feature_names.pkl")