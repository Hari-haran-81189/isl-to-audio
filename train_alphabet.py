
import pandas as pd
import os
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------
# 1. LOAD DATASET
# ---------------------------------

CSV_FILE = "dataset/alphabet/alphabet_landmarks.csv"

df = pd.read_csv(CSV_FILE)

print("Dataset loaded successfully!")
print("Total samples:", len(df))

# ---------------------------------
# 2. PREPARE FEATURES AND LABELS
# ---------------------------------

metadata = ["label", "recording_id", "frame_number"]

feature_columns = [
    col for col in df.columns if col not in metadata
]

X = df[feature_columns]
y = df["label"]

# ---------------------------------
# 3. SPLIT BY RECORDING PER LETTER
# ---------------------------------

rng = np.random.RandomState(42)

train_indices = []
test_indices = []

for letter in sorted(df["label"].unique()):

    letter_data = df[df["label"] == letter]

    recordings = letter_data["recording_id"].unique().copy()

    rng.shuffle(recordings)

    if len(recordings) < 2:
        raise ValueError(
            f"Not enough recordings for letter {letter}"
        )

    # Keep one complete recording for testing
    test_recording = recordings[0]

    test_indices.extend(
        letter_data.index[
            letter_data["recording_id"] == test_recording
        ].tolist()
    )

    train_indices.extend(
        letter_data.index[
            letter_data["recording_id"] != test_recording
        ].tolist()
    )

X_train = X.loc[train_indices]
X_test = X.loc[test_indices]

y_train = y.loc[train_indices]
y_test = y.loc[test_indices]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Test letters:", sorted(y_test.unique()))

# ---------------------------------
# 4. CREATE RANDOM FOREST
# ---------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# ---------------------------------
# 5. TRAIN MODEL
# ---------------------------------

print("\nTraining started...")

model.fit(X_train, y_train)

print("Training completed!")

# ---------------------------------
# 6. EVALUATE MODEL
# ---------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL RESULTS ==========")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=sorted(df["label"].unique()),
        zero_division=0
    )
)

# ---------------------------------
# 7. TRAIN FINAL MODEL ON ALL DATA
# ---------------------------------

print("\nTraining final model using all samples...")

final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

final_model.fit(X, y)

# ---------------------------------
# 8. SAVE FINAL MODEL
# ---------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": final_model,
        "feature_columns": feature_columns
    },
    "models/alphabet_model.joblib"
)

print("\nFinal model saved successfully!")
print("Location: models/alphabet_model.joblib")
