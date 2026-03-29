# NIKHIL AGARWAL-AIML-project
# Student Performance Predictor

A simple machine learning project that predicts student marks using **Linear Regression**.

## Project Idea
Predict marks based on:
- Study hours ( amount of student spent while studying )
- Sleep time  ( amount of time student invested during rest )
- Attendance  ( total number of class in which student was present )

## Concepts Used

- Linear Regression
- Feature-based prediction
- Train-test split
- Model evaluation (MAE, R2 score)

## Tech Stack

- Python
- Pandas
- Scikit-learn

## Input and Output

- **Input:** Hours studied, sleep time, attendance percentage
- **Output:** Predicted marks out of 100

## Files

- `student_performance_predictor.py` - Main script for training and prediction

## Installation

Install dependencies:

```bash
pip install pandas scikit-learn
```

## Run the Project

```bash
python student_performance_predictor.py
```

## How It Works

1. Builds a sample containing student's dataset
2. Trains a Linear Regression model
3. Evaluates model performance using MAE and R2 score
4. Takes user input for new student data
5. Predicts and displays marks

## Example Input

- Study hours per day: 6
- Sleep hours per day: 7
- Attendance percentage: 85

## Example Output

- Predicted Marks: around 75-85 (depends on trained model)

## Learning Outcomes

This project helps you practice:
- Basic machine learning workflow
- Data handling with Pandas
- Regression modeling with Scikit-learn
- User input validation in Python

---

hon
- Structured data handling using Pandas
