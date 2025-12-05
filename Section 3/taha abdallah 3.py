Open In Colab

# taha abdallah- Task3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from google.colab import files

uploaded = files.upload()
file_name = list(uploaded.keys())[0]

df = pd.read_csv(file_name)
print("Preview of dataset:")
print(df.head())

feature_cols = ['Age', 'Weight', 'Height', 'Duration', 'Heart_Rate']
X = df[feature_cols]
y = df['Calories']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nModel Performance:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

new_day = pd.DataFrame([{
    'Age': 25,
    'Weight': 70,
    'Height': 175,
    'Duration': 60,
    'Heart_Rate': 130
}])

new_day_scaled = scaler.transform(new_day)
pred_calories = model.predict(new_day_scaled)
print("\nPredicted Calories Burned:", round(pred_calories[0], 2))
# taha abdallah - Task3
     
Upload widget is only available when the cell has been executed in the current browser session. Please rerun this cell to enable.
Saving calories.csv to calories (1).csv
Preview of dataset:
    User_ID  Gender  Age  Height  Weight  Duration  Heart_Rate  Body_Temp  \
0  14733363    male   68   190.0    94.0      29.0       105.0       40.8   
1  14861698  female   20   166.0    60.0      14.0        94.0       40.3   
2  11179863    male   69   179.0    79.0       5.0        88.0       38.7   
3  16180408  female   34   179.0    71.0      13.0       100.0       40.5   
4  17771927  female   27   154.0    58.0      10.0        81.0       39.8   

   Calories  
0     231.0  
1      66.0  
2      26.0  
3      71.0  
4      35.0  

Model Performance:
Mean Squared Error: 165.53666705611326
R2 Score: 0.9589828493404986

Predicted Calories Burned: 378.97