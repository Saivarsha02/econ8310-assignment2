import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load training data
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")

# Drop missing values
data = data.dropna()

# Remove unnecessary columns
if 'id' in data.columns:
    data = data.drop(columns=['id'])
if 'DateTime' in data.columns:
    data = data.drop(columns=['DateTime'])

# Identify categorical columns and encode them
categorical_cols = data.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store encoders for test data

# Define target and features
y = data['meal']
X = data.drop(columns=['meal'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree Model
dt_model = DecisionTreeClassifier(max_depth=5)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print("Decision Tree Accuracy:", dt_acc)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_acc)

# Boosted Tree Model
xgb_model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.5)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
print("XGBoost Accuracy:", xgb_acc)

# Load test data for final predictions
test_data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")

# Drop unnecessary columns from test data
if 'id' in test_data.columns:
    test_data = test_data.drop(columns=['id'])
if 'DateTime' in test_data.columns:
    test_data = test_data.drop(columns=['DateTime'])

# Apply encoding to test data safely
for col in categorical_cols:
    if col in test_data.columns:
        test_data[col] = test_data[col].map(lambda s: label_encoders[col].classes_.tolist().index(s) if s in label_encoders[col].classes_ else -1)

# Ensure test data columns match training data
test_data = test_data[X.columns]  # Reorder columns to match training data

# Choose best model based on highest accuracy
best_model = max([(dt_model, dt_acc), (rf_model, rf_acc), (xgb_model, xgb_acc)], key=lambda x: x[1])[0]
pred = best_model.predict(test_data)

# Convert predictions to binary format (1 or 0)
pred = [int(p) for p in pred]

# Save the model and predictions
joblib.dump(best_model, 'modelFit.pkl')  # Save best model
pd.DataFrame(pred, columns=['meal_prediction']).to_csv('predictions.csv', index=False)
