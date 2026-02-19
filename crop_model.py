import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset (tab-separated fix)
data = pd.read_csv("data/Crop_recommendation.csv", sep="\t")

print("Columns:", data.columns)

X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

pickle.dump(model, open("models/crop_model.pkl", "wb"))

print("Model saved successfully!")
