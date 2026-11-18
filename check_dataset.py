
import pandas as pd

file_path = "dataset/alphabet/alphabet_landmarks.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET OVERVIEW ==========")

print("Total rows:", len(df))
print("Total columns:", len(df.columns))

print("\n========== LETTER COUNTS ==========")

print(df["label"].value_counts().sort_index())

print("\n========== RECORDING COUNTS ==========")

print(
    df.groupby("label")["recording_id"].nunique()
)

print("\n========== HAND DETECTION ==========")

print(
    df[["left_present", "right_present"]].sum()
)

print("\n========== FIRST 5 ROWS ==========")

print(df.head())

print("\n========== MISSING VALUES ==========")

print(df.isnull().sum().sum())

print("\n========== HAND COMBINATIONS ==========")

both = (
    (df["left_present"] == 1) &
    (df["right_present"] == 1)
).sum()

left_only = (
    (df["left_present"] == 1) &
    (df["right_present"] == 0)
).sum()

right_only = (
    (df["left_present"] == 0) &
    (df["right_present"] == 1)
).sum()

print("Both hands:", both)
print("Left hand only:", left_only)
print("Right hand only:", right_only)
