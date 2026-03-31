import numpy as np
from sklearn.neural_network import MLPClassifier
import pandas as pd

# Load data
data = pd.read_csv("dataset.csv")
X = data.drop("disease", axis=1)
y = data["disease"]

model = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=500)
model.fit(X, y)

# Take user input
print("Enter symptoms (1 or 0):")
fever = int(input("Fever: "))
cough = int(input("Cough: "))
headache = int(input("Headache: "))
fatigue = int(input("Fatigue: "))

input_data = np.array([[fever, cough, headache, fatigue]])

prediction = model.predict(input_data)

print("Predicted Disease:", prediction[0])
