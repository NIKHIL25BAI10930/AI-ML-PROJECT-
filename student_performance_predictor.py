"""
Student Performance Predictor
Predict marks using Linear Regression based on:
- studhours
- sleephours
- attndancepercent
"""

from __future__ import annotations

import pandas as pan
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def builddataset() -> pan.DataFrame:
            
            data = {
                "studhours": [1, 2, 3, 4, 5, 6, 7, 8, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
                "sleephours": [5, 6, 6, 7, 7, 8, 8, 7, 5.5, 6.5, 7, 7.5, 8, 6.5],
                "attndancepercent": [60, 65, 70, 75, 80, 85, 90, 95, 68, 73, 78, 83, 88, 92],
                "marks": [35, 42, 50, 58, 65, 73, 82, 90, 46, 54, 61, 69, 78, 85],
            }
            return pan.DataFrame(data)


def traimodel(df: pan.DataFrame) -> LinearRegression:


    
    features = ["studhours", "sleephours", "attndancepercent"]
    target = "marks"
    X = df[features]

    
    y = df[target]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()


    
    model.fit(Xtrain, ytrain)

    predictions = model.predict(Xtest)
    
    mae = mean_absolute_error(ytest, predictions)
    r2 = r2_score(ytest, predictions)

    print("\nModel trained successfully.")
    print(f"MAE: {mae:.2f}")
    
    print(f"R² Score: {r2:.2f}")
    return model


def getfloatinput(prompt: str, min_value: float, max_value: float) -> float:
           
            while True:
                value = input(prompt).strip()
                try:
                    num = float(value)
                    
                    if min_value <= num <= max_value:
                        return num
                        
                    print("Enter a value between", {min_value}," and", {max_value})
                except ValueError:
                    
                    print("Please enter a valid number")


def predictmarks(model: LinearRegression) -> None:
    
    print("\nEnter student details to predict marks:")
    
    studhours = getfloatinput("Study hours per day (0-12): ", 0, 12)
    
    sleephours = getfloatinput("Sleep hours per day (0-12): ", 0, 12)
    attndancepercent = getfloatinput("Attendance percentage (0-100): ", 0, 100)

    input_df = pan.DataFrame(
        {
            "study hours": [studhours],
            "sleep hours": [sleephours],
            "attendance percent": [attndancepercent],
        }
    )

    predicted_marks = model.predict(input_df)[0]
    predicted_marks = max(0, min(100, predicted_marks))
    print(f"\nPredicted Marks: {predicted_marks:.2f}/100")


def main() -> None:
        print("=" * 50)
        print("Student Performance Predictor")
        print("=" * 50)
    
        df = builddataset()
        model = traimodel(df)
    
        while True:
            predictmarks(model)
            again = input("\nPredict again? (yes/no): ").strip().lower()
            if again not in {"yes", "y"}:
                print("Good luck with your studies!")
                break


if __name__ == "__main__":
    main()
