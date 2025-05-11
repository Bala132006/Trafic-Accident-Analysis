from google.colab import files


from google.colab import files
import pandas as pd

uploaded = files.upload() # This will prompt you to upload the file
# Once uploaded, the file will be available in the current working directory

df = pd.read_csv('RTA Dataset.csv')



df.info()
df.head()



df.isnull().sum()
df.duplicated().sum()




import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=df, x='Accident_severity')
plt.title('Distribution of Accident Severity')
plt.show()





sns.boxplot(x='Accident_severity', y='Number_of_casualties', data=df)







X = df.drop('Accident_severity', axis=1)
y = df['Accident_severity']





from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in X.select_dtypes(include='object').columns:
X[col] = le.fit_transform(X[col].astype(str))





X = pd.get_dummies(X, drop_first=True)






from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)





from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)




from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)






from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)








sample_input = X.iloc[0]
prediction = model.predict([sample_input])







sample_df = pd.DataFrame([sample_input])
sample_df_encoded = scaler.transform(sample_df)






final_prediction = model.predict(sample_df_encoded)
print("Predicted Severity:", final_prediction[0])






!pip install gradio
import gradio as gr



def predict_severity(Time, Day_of_week, **kwargs): # all necessary inputs
# Create dataframe, encode, scale, predict
return "Predicted Severity"







import gradio as gr

def predict_severity(Time, Day_of_week, Number_of_vehicles_involved, Number_of_casualties,
Road_surface_conditions, Light_conditions, Weather_conditions,
Type_of_collision, Pedestrian_movement,Driving_experience,
Vehicle_driver_relation,Lanes_or_Medians,Types_of_Junction,
Vehicle_movement):
# Create dataframe for the single input
input_data = pd.DataFrame([[Time, Day_of_week, Number_of_vehicles_involved, Number_of_casualties,
Road_surface_conditions, Light_conditions, Weather_conditions,
Type_of_collision, Pedestrian_movement,Driving_experience,
Vehicle_driver_relation,Lanes_or_Medians,Types_of_Junction,
Vehicle_movement]],
columns=['Time', 'Day_of_week', 'Number_of_vehicles_involved', 'Number_of_casualties',
'Road_surface_conditions', 'Light_conditions', 'Weather_conditions',
'Type_of_collision', 'Pedestrian_movement','Driving_experience',
'Vehicle_driver_relation','Lanes_or_Medians','Types_of_Junction',
'Vehicle_movement'])
import gradio as gr
import pandas as pd

def predict_severity(Time, Day_of_week, Number_of_vehicles_involved, Number_of_casualties,
Road_surface_conditions, Light_conditions, Weather_conditions,
Type_of_collision, Pedestrian_movement,Driving_experience,
Vehicle_driver_relation,Lanes_or_Medians,Types_of_Junction,
Vehicle_movement):
# Create dataframe for the single input
input_data = pd.DataFrame([[Time, Day_of_week, Number_of_vehicles_involved, Number_of_casualties,
Road_surface_conditions, Light_conditions, Weather_conditions,
Type_of_collision, Pedestrian_movement,Driving_experience,
Vehicle_driver_relation,Lanes_or_Medians,Types_of_Junction,
Vehicle_movement]],
columns=['Time', 'Day_of_week', 'Number_of_vehicles_involved', 'Number_of_casualties',
'Road_surface_conditions', 'Light_conditions', 'Weather_conditions',
'Type_of_collision', 'Pedestrian_movement','Driving_experience',
'Vehicle_driver_relation','Lanes_or_Medians','Types_of_Junction',
'Vehicle_movement'])

# Get dummies and ensure all columns from original X are present
input_data = pd.get_dummies(input_data)














import gradio as gr
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Example model setup (load your trained model in a real case)
model = RandomForestClassifier()
# Assume model is trained already with `X_train`, `y_train`
# Also assume `le_dict` and `scaler` are saved from training

# Dummy label encoder and scaler setup (replace with actual trained ones)
le_dict = {} # Dictionary of LabelEncoders per column
scaler = StandardScaler()

# Prediction function
def predict_accident_severity(
age_band, sex, education, driving_exp, vehicle_type, weather, road_surface, light
):
input_data = pd.DataFrame([{
"Age_band_of_driver": age_band,
"Sex_of_driver": sex,
"Educational_level": education,
"Driving_experience": driving_exp,
"Type_of_vehicle": vehicle_type,
"Weather_conditions": weather,
"Road_surface_conditions": road_surface,
"Light_conditions": light
}])

# Encode input (assuming label encoders are available for each feature)
for col in input_data.columns:
if col not in le_dict:
le = LabelEncoder()
input_data[col] = le.fit_transform(input_data[col])
le_dict[col] = le
else:
input_data[col] = le_dict[col].transform(input_data[col])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)[0]
return f"🚨 Predicted Accident Severity: {prediction}"

# Interface setup
interface = gr.Interface(
fn=predict_accident_severity,
inputs=[
gr.Dropdown(["Under 18", "18-30", "31-50", "Over 51"], label="Driver Age Band"),
gr.Radio(["Male", "Female"], label="Sex of Driver"),
gr.Dropdown(["No Education", "Primary", "Junior High", "High School", "Above High School"], label="Education Level"),
gr.Dropdown(["No Experience", "1-2yr", "2-5yr", "5-10yr", "Above 10yr"], label="Driving Experience"),
gr.Dropdown(["Automobile", "Lorry", "Public (>45 seats)", "Taxi"], label="Type of Vehicle"),
gr.Dropdown(["Clear", "Rainy", "Fog", "Windy", "Dust Storm"], label="Weather Conditions"),
gr.Dropdown(["Dry", "Wet", "Snow", "Flooded"], label="Road Surface Conditions"),
gr.Dropdown(["Daylight", "Dark (No lights)", "Dark (Lights On)"], label="Light Conditions")
],
outputs="text",
title="🚗 Road Accident Severity Predictor",
description="Predicts the severity of a traffic accident based on input conditions using AI-driven analysis."
)

interface.launch()









