import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("loan_data.csv")

print("Dataset Loaded Successfully")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

df.fillna(df.mode().iloc[0], inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

le = LabelEncoder()

for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])

print("\nData After Encoding:")
print(df.head())

X = df.drop("Loan_Status", axis=1)

y = df["Loan_Status"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel Trained Successfully")

y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred[:10])

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.show()

importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

print("\nFeature Importance:")
print(importance.sort_values(
    by='Coefficient',
    ascending=False
))
