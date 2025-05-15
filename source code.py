pip install pandas scikit-learn gradio

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import gradio as gr

# Load and preprocess data
df = pd.read_csv("traffic_accidents.csv")

# Feature selection (assuming these columns exist in the dataset)
features = ['weather', 'time_of_day', 'road_type', 'speed']
target = 'accident_occurred'

# Handling missing values
df.dropna(subset=features + [target], inplace=True)

# Encode categorical variables
df['weather'] = df['weather'].map({'Clear': 0, 'Rain': 1, 'Fog': 2})
df['road_type'] = df['road_type'].map({'Urban': 0, 'Highway': 1, 'Rural': 2})
df['time_of_day'] = df['time_of_day'].map({'Day': 0, 'Night': 1})

# Prepare X and y for model training
X = df[features]
y = df[target]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a RandomForestClassifier model
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Evaluate the model
accuracy = model.score(X_test_scaled, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Define prediction function
def predict_accident(weather, time_of_day, road_type, speed):
    # Map input values to their encoded values
    weather_map = {'Clear': 0, 'Rain': 1, 'Fog': 2}
    time_of_day_map = {'Day': 0, 'Night': 1}
    road_type_map = {'Urban': 0, 'Highway': 1, 'Rural': 2}
    
    # Encode inputs
    weather_encoded = weather_map[weather]
    time_of_day_encoded = time_of_day_map[time_of_day]
    road_type_encoded = road_type_map[road_type]

    # Prepare input data (same features as the training data)
    input_data = pd.DataFrame([[weather_encoded, time_of_day_encoded, road_type_encoded, speed]], 
                              columns=['weather', 'time_of_day', 'road_type', 'speed'])

    # Scale the input features using the same scaler used during training
    input_scaled = scaler.transform(input_data)

    # Get the prediction from the model
    prediction = model.predict(input_scaled)

    # Return the prediction result
    return "Accident Occurred" if prediction[0] == 1 else "No Accident"

# Define Gradio inputs
inputs = [
    gr.Radio(choices=["Clear", "Rain", "Fog"], label="Weather"),
    gr.Radio(choices=["Day", "Night"], label="Time of Day"),
    gr.Radio(choices=["Urban", "Highway", "Rural"], label="Road Type"),
    gr.Slider(minimum=0, maximum=120, label="Speed (km/h)", step=30)
]

# Define Gradio output
outputs = gr.Textbox(label="Prediction Result")

# Create Gradio interface
iface = gr.Interface(fn=predict_accident, inputs=inputs, outputs=outputs, live=True)

# Launch the interface
iface.launch()
