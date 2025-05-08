from google.colab import files
import pandas as pd

df = pd.read_csv("accident_prediction_india.csv")
df.head()
df.info()
df.describe()
df.shape
df.columns
print("Missing values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=df, x='Accident Severity')
plt.title("Accident Severity Distribution")
plt.show()

sns.histplot(df['Number of Vehicles Involved'], kde=True)
plt.title("Vehicles Involved Distribution")
plt.show()
target = 'Accident Severity'
X = df.drop(columns=[target])
y = df[target]
X = pd.get_dummies(X, drop_first=True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
sample_input = [X_test[0]]  # Or any row from your test set
print("Prediction:", model.predict(sample_input))
from sklearn.ensemble import RandomForestClassifier
!pip install gradio
import gradio as gr
def predict_accident(State_Name, City_Name, Year, Month, Day_of_Week, Time_of_Day,
                     Number_of_Vehicles, Vehicle_Type, Casualties, Fatalities,
                     Weather, Road_Type, Road_Condition, Lighting, Traffic_Control,
                     Speed_Limit, Driver_Age, Driver_Gender, License_Status,
                     Alcohol, Location):

    input_dict = {
        'State Name': State_Name,
        'City Name': City_Name,
        'Year': int(Year),
        'Month': Month,
        'Day of Week': Day_of_Week,
        'Time of Day': Time_of_Day,
        'Number of Vehicles Involved': int(Number_of_Vehicles),
        'Vehicle Type Involved': Vehicle_Type,
        'Number of Casualties': int(Casualties),
        'Number of Fatalities': int(Fatalities),
        'Weather Conditions': Weather,
        'Road Type': Road_Type,
        'Road Condition': Road_Condition,
        'Lighting Conditions': Lighting,
        'Traffic Control Presence': Traffic_Control,
        'Speed Limit (km/h)': int(Speed_Limit),
        'Driver Age': int(Driver_Age),
        'Driver Gender': Driver_Gender,
        'Driver License Status': License_Status,
        'Alcohol Involvement': Alcohol,
        'Accident Location Details': Location
    }

    df_input = pd.DataFrame([input_dict])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=X.columns, fill_value=0)
    input_scaled = scaler.transform(df_input)
    pred = model.predict(input_scaled)
    return f"Predicted Accident Severity: {pred[0]}"
interface = gr.Interface(
    fn=predict_accident,inputs=[
        gr.Textbox(label="State Name"),
        gr.Textbox(label="City Name"),
        gr.Textbox(label="Year"),
        gr.Textbox(label="Month"),
        gr.Textbox(label="Day of Week"),
        gr.Textbox(label="Time of Day"),
        gr.Textbox(label="Number of Vehicles Involved"),
        gr.Textbox(label="Vehicle Type Involved"),
        gr.Textbox(label="Number of Casualties"),
        gr.Textbox(label="Number of Fatalities"),
        gr.Textbox(label="Weather Conditions"),
        gr.Textbox(label="Road Type"),
        gr.Textbox(label="Road Condition"),
        gr.Textbox(label="Lighting Conditions"),
        gr.Textbox(label="Traffic Control Presence"),
        gr.Textbox(label="Speed Limit (km/h)"),
        gr.Textbox(label="Driver Age"),
        gr.Textbox(label="Driver Gender"),
        gr.Textbox(label="Driver License Status"),
        gr.Textbox(label="Alcohol Involvement"),
        gr.Textbox(label="Accident Location Details"),
    ],
    outputs="text",
    title="Accident Severity Prediction App"
)

interface.launch()
