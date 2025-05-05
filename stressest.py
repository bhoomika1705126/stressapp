# stress_model.ipynb

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv(r'C:\Users\hp\Desktop\project ml\dailyActivity_merged_cleaned.csv')

# Select required features
df = df[['TotalSteps', 'Calories', 'VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes']]

# Define synthetic stress level formula (activity → lower stress)
def generate_stress(row):
    active_score = row['VeryActiveMinutes'] * 2 + row['FairlyActiveMinutes'] * 1.5 + row['LightlyActiveMinutes'] * 1
    calorie_score = row['Calories'] / 10
    step_score = row['TotalSteps'] / 100
    activity_score = active_score + calorie_score + step_score
    stress = 10 - (activity_score / 100)
    return max(0, min(10, stress))

df['StressLevel'] = df.apply(generate_stress, axis=1)

# Split and train model
X = df[['TotalSteps', 'Calories', 'VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes']]
y = df['StressLevel']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'stress_model.pkl')

print("Model trained and saved successfully!")
# stress_app.py

import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('stress_model.pkl')

st.title("Stress Level Estimator based on Daily Activity")

st.write("Move the sliders to simulate your daily activity.")

# Sliders for input
steps = st.slider('Total Steps', 0, 30000, 8000)
calories = st.slider('Calories Burned', 0, 5000, 2200)
very_active = st.slider('Very Active Minutes', 0, 180, 30)
fairly_active = st.slider('Fairly Active Minutes', 0, 180, 20)
lightly_active = st.slider('Lightly Active Minutes', 0, 300, 60)

# Predict stress level
if st.button('Predict Stress Level'):
    input_data = np.array([[steps, calories, very_active, fairly_active, lightly_active]])
    stress_level = model.predict(input_data)[0]
    stress_level = round(stress_level, 2)
    
    st.success(f"Predicted Stress Level: {stress_level} / 10")

    if stress_level > 6:
        st.warning("⚠️ High Stress. Consider more physical activity or breaks.")
    elif stress_level < 3:
        st.info("✅ Low Stress. Keep up the healthy routine!")
    else:
        st.write("🧘 Moderate Stress. Balance your activity and rest.")