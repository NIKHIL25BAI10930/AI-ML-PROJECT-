"""
Student Performance Predictor
Predict marks using Linear Regression based on:
- study_hours
- sleep_hours
- attendance_percent
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def build_sample_dataset() -> pd.DataFrame:
    """Create a small sample dataset for demonstration."""
    data = {
        "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
        "sleep_hours": [5, 6, 6, 7, 7, 8, 8, 7, 5.5, 6.5, 7, 7.5, 8, 6.5],
        "attendance_percent": [60, 65, 70, 75, 80, 85, 90, 95, 68, 73, 78, 83, 88, 92],
        "marks": [35, 42, 50, 58, 65, 73, 82, 90, 46, 54, 61, 69, 78, 85],
    }
    return pd.DataFrame(data)


def train_model(df: pd.DataFrame) -> LinearRegression:
    """Train and evaluate a Linear Regression model."""
    features = ["study_hours", "sleep_hours", "attendance_percent"]
    target = "marks"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nModel trained successfully.")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")
    return model


def get_float_input(prompt: str, min_value: float, max_value: float) -> float:
    """Get validated float input from user."""
    while True:
        value = input(prompt).strip()
        try:
            num = float(value)
            if min_value <= num <= max_value:
                return num
            print(f"Enter a value between {min_value} and {max_value}.")
        except ValueError:
            print("Please enter a valid number.")


def predict_marks(model: LinearRegression) -> None:
    """Take user input and predict marks."""
    print("\nEnter student details to predict marks:")
    study_hours = get_float_input("Study hours per day (0-12): ", 0, 12)
    sleep_hours = get_float_input("Sleep hours per day (0-12): ", 0, 12)
    attendance_percent = get_float_input("Attendance percentage (0-100): ", 0, 100)

    input_df = pd.DataFrame(
        {
            "study_hours": [study_hours],
            "sleep_hours": [sleep_hours],
            "attendance_percent": [attendance_percent],
        }
    )

    predicted_marks = model.predict(input_df)[0]
    predicted_marks = max(0, min(100, predicted_marks))
    print(f"\nPredicted Marks: {predicted_marks:.2f}/100")


def main() -> None:
    print("=" * 50)
    print("Student Performance Predictor")
    print("=" * 50)

    df = build_sample_dataset()
    model = train_model(df)

    while True:
        predict_marks(model)
        again = input("\nPredict again? (yes/no): ").strip().lower()
        if again not in {"yes", "y"}:
            print("Good luck with your studies!")
            break


if __name__ == "__main__":
    main()
