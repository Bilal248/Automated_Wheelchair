import joblib
import pandas as pd
import sys

model = joblib.load('rf_model.pkl')
le = joblib.load('label_encoder.pkl')

input_data = [float(val) for val in sys.argv[1:7]]
pred = model.predict([input_data])[0]
print(le.inverse_transform([pred])[0])