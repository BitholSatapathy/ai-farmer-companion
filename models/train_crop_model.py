import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load dataset
data = pd.read_csv("../data/Crop_recommendation.csv")

# 2. Separate features (X) and label (y)
X = data.drop("label", axis=1)
y = data["label"]

# 3. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5. Train model
model.fit(X_train, y_train)

# 6. Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# 7. Save model
with open("crop_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")