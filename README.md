# NIKHIL AGARWAL-AIML-project
# Student Performance Predictor

A simple machine learning project that predicts student marks using **Linear Regression**.

## Project Idea
Predict marks based on:
- Study hours
- Sleep time
- Attendance

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

1. Builds a sample student dataset
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

# Movie Recommendation System

A content-based recommendation project that suggests movies based on user interests.

## Project Idea

Recommend movies based on:
- User interest keywords
- Genre preferences
- Content similarity

## Concepts Used

- Recommendation systems
- Content-based filtering
- Cosine similarity
- Text tokenization for matching

## Tech Stack

- Python
- Pandas

## Input and Output

- **Input:** User interests (for example: `sci-fi space thriller`)
- **Output:** Top recommended movies with match scores

## Files

- `movie_recommendation_system.py` - Main script for content-based movie recommendations

## Run the Project

```bash
python movie_recommendation_system.py
```

## How It Works

1. Creates a sample movie dataset with genres and keywords
2. Converts user interests and movie content into normalized tokens
3. Calculates similarity score between user profile and each movie
4. Sorts movies by similarity (and rating as tie-breaker)
5. Displays top recommendations

## Example Input

- `sci-fi thriller mind-bending`

## Example Output

- Inception
- The Matrix
- Interstellar

## Learning Outcomes

This project helps you practice:
- Building a basic recommendation engine
- Content-based filtering logic
- Similarity scoring with Python
- Structured data handling using Pandas
