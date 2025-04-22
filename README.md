# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load dataset and preprocess features (handle missing values, encode categories).
2. Divide data into training and testing sets.
3. Fit a Decision Tree Classifier on the training set using entropy or gini.
4. Predict & Evaluate: Predict on test set and calculate accuracy or visualize the tree.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Simon Malachi S
RegisterNumber:  212224040318
*/

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("C:/Users/YourUsername/Downloads/Employee.csv")  # <-- Update path as needed

# Encode 'salary' column
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

# Feature selection
x = data[[
    "satisfaction_level", 
    "last_evaluation", 
    "number_project", 
    "average_montly_hours", 
    "time_spend_company", 
    "Work_accident", 
    "promotion_last_5years", 
    "salary"
]]
y = data["left"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Train Decision Tree
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)

# Evaluate
y_pred = dt.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict using feature names to avoid warning
sample_input = pd.DataFrame([[0.5, 0.8, 9, 260, 6, 0, 1, 2]], columns=x.columns)
custom_prediction = dt.predict(sample_input)
print("Custom Prediction (0 = Not Left, 1 = Left):", custom_prediction[0])

# Plot the tree
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=x.columns, class_names=["Not Left", "Left"], filled=True)
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/074f81d6-dd8a-4c35-9228-e0882191ed97)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
