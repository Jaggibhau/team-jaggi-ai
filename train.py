
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Clean BIG/SMALL sequence
data = [
    "SMALL", "BIG", "SMALL", "BIG", "BIG", "SMALL", "SMALL", "BIG",
    "BIG", "SMALL", "BIG", "SMALL", "BIG", "BIG", "SMALL", "SMALL",
    "SMALL", "SMALL", "BIG", "SMALL", "BIG", "SMALL", "BIG", "BIG",
    "BIG", "SMALL", "BIG", "SMALL", "BIG", "SMALL", "BIG", "BIG",
    "SMALL", "SMALL", "SMALL", "BIG", "BIG", "BIG", "SMALL", "SMALL",
    "BIG", "SMALL", "BIG", "BIG", "SMALL", "SMALL", "BIG", "BIG",
    "BIG", "BIG", "BIG", "SMALL", "BIG", "BIG", "SMALL", "SMALL",
    "BIG", "SMALL", "BIG", "BIG", "BIG", "SMALL", "SMALL", "BIG",
    "SMALL", "BIG", "SMALL", "SMALL", "SMALL", "BIG", "BIG", "SMALL",
    "SMALL", "BIG", "BIG", "BIG", "BIG", "SMALL", "BIG", "BIG",
    "BIG", "SMALL", "BIG", "BIG", "SMALL", "BIG", "BIG", "BIG",
    "SMALL", "BIG", "SMALL", "SMALL", "BIG", "BIG", "SMALL", "BIG"
]

# Encode: BIG=1, SMALL=0
mapping = {"BIG": 1, "SMALL": 0}
labels = [mapping[x] for x in data]

# Create sequences of 3 to predict the 4th
X, y = [], []
for i in range(3, len(labels)):
    X.append(labels[i-3:i])
    y.append(labels[i])

# Train the model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the model
with open("model/decision_tree_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully.")
