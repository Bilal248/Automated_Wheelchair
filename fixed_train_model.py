
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv('uni7_data.csv')

# Drop rows with NaN bearing or direction
df = df.dropna(subset=['bearing', 'direction'])

# Features and labels
X = df[['timeOfFlight', 'hc', 'latitude', 'longitude', 'speed', 'bearing']]
y = df['direction']

# Encode direction labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

# Evaluate and print classification report
y_pred = model.predict(X)
print(classification_report(y_encoded, y_pred, target_names=le.classes_))

# Save the model and label encoder
joblib.dump(model, 'rf_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Model and label encoder saved.")
