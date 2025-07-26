# Titanic Survival Analysis using Python

This project analyzes Titanic passenger data to understand what factors helped people survive. It includes data cleaning, graphs, and machine learning.

## Files in the Project

- `titanic.csv` – Dataset  
- `titanic.py` – Python code  
- `titanic_eda/` – Folder with graphs like age, fare, survival, etc.

## What is Inside the Data?

- `Survived` – 0 = No, 1 = Yes  
- `Pclass` – Class (1 = 1st, 2 = 2nd, 3 = 3rd)  
- `Sex`, `Age`, `Fare`, `SibSp`, `Parch` – Other passenger info

## Graphs and EDA

Graphs are stored in the `titanic_eda` folder:
- Age, Fare, Sex, Survival graphs  
- Heatmap to show relationships between columns

## Machine Learning

Used **Random Forest** to predict survival based on input features.

## ▶️ How to Run

1. Open terminal  
2. Run:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python titanic.py
