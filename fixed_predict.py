
import joblib
import sys

if len(sys.argv) != 7:
    print("Error: Expected 6 input features: TOF, HC, Latitude, Longitude, Speed, Bearing")
    sys.exit(1)

# Parse input features
tof = float(sys.argv[1])
hc = float(sys.argv[2])
lat = float(sys.argv[3])
lon = float(sys.argv[4])
speed = float(sys.argv[5])
bearing = float(sys.argv[6])

# Load model and encoder
model = joblib.load('rf_model.pkl')
le = joblib.load('label_encoder.pkl')

# Make prediction
X = [[tof, hc, lat, lon, speed, bearing]]
pred = model.predict(X)[0]
label = le.inverse_transform([pred])[0]

print(f"Predicted Direction: {label}")
