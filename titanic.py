import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
def sanitize_filename(s):
    return re.sub(r'[^a-zA-Z0-9_]', '_', s)
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df
def preprocess(df):
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    return df
def eda(df, output_dir="titanic_eda"):
    os.makedirs(output_dir, exist_ok=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Histogram of {col}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{sanitize_filename(col)}_hist.png")
        plt.close()
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()
def model_train(df):
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n Accuracy: {acc:.2f}")
    print("\n Classification Report:\n", classification_report(y_test, y_pred))
    print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n Top Features:\n", feature_imp.head())
def main():
    filepath = "titanic.csv"  # Change if needed
    df = load_data(filepath)
    print(" Dataset loaded successfully.")

    df = preprocess(df)
    print("Preprocessing done.")

    eda(df)
    print(" EDA charts saved to 'titanic_eda/' folder.")

    model_train(df)

if __name__ == "__main__":
    main()
