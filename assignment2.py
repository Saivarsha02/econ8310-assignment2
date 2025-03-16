import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib

# Load training data
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
data = pd.read_csv(train_url)

# Define target variable and features
y = data['meal']
X = data.drop(columns=['meal', 'id', 'DateTime'])  # Remove non-numeric columns
X = pd.get_dummies(X, drop_first=True)  # Convert categorical features if any

# Split dataset into training and testing sets
x, xt, y, yt = train_test_split(X, y, test_size=0.1, random_state=42)

# Define the best-performing model
model = XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.2, random_state=42, objective='binary:logistic')

# Train the model
modelFit = model.fit(x, y)

# Load test data
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
test_data = pd.read_csv(test_url)
test_data = test_data.drop(columns=['id', 'DateTime'])  # Drop non-numeric columns
test_data = pd.get_dummies(test_data, drop_first=True)
test_data = test_data.reindex(columns=X.columns, fill_value=0)  # Ensure same columns

# Make predictions using the trained model
pred = modelFit.predict(test_data)
pred = [int(p) for p in pred]  # Convert to required format

# Save predictions
pd.DataFrame(pred, columns=["meal_prediction"]).to_csv("predictions.csv", index=False)

# Save the trained model
joblib.dump(modelFit, "modelFit.pkl")

print("Best model selected and predictions are saved successfully.")