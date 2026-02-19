import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
data = pd.read_csv("data/yield_data.csv")

print("Columns:", data.columns)

# Simple feature selection (numeric only)
data = data[['Year', 'Value']]

# X and y
X = data[['Year']]
y = data['Value']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Score
score = model.score(X_test, y_test)
print("Model R2 Score:", score)

# Save model
pickle.dump(model, open("models/yield_model.pkl", "wb"))

print("Yield model saved successfully!")
