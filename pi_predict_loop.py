# pi_predict_loop.py

import serial
import joblib
import numpy as np
import time

# Load your trained model
model = joblib.load("rf_model.pkl")
le = joblib.load("label_encoder.pkl")

# Set up serial connection to Arduino
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
ser.flush()

def preprocess(tof, hc, lat, lon, speed, bearing):
    # Add normalization if needed
    return np.array([[float(tof), float(hc), float(lat), float(lon), float(speed), float(bearing)]])

while True:
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            print(f"Received: {line}")
            vals = line.split(',')

            if len(vals) >= 6:
                features = preprocess(*vals[:6])
                prediction = model.predict(features)
                direction = le.inverse_transform(prediction)[0]
                print(f"AI Decision: {direction}")

                # Send decision back to Arduino if needed
                ser.write((direction + "\n").encode('utf-8'))
        except Exception as e:
            print(f"Error: {e}")
    
    time.sleep(0.1)
