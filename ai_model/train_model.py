import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv('uni7_data.csv')
df.dropna(subset=['direction'], inplace=True)

le = LabelEncoder()
df['direction_encoded'] = le.fit_transform(df['direction'])

X = df[['timeOfFlight', 'hc', 'latitude', 'longitude', 'speed', 'bearing']]
y = df['direction_encoded']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'rf_model.pkl')
joblib.dump(le, 'label_encoder.pkl')