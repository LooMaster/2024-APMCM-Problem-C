import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    'Year': [2019, 2020, 2021, 2022, 2023],
    'Sales_Forecast': [93.9, 102.6, 114.5, 123.6, 133.90],
    'Average_Price_Forecast': [2.26, 2.40, 2.55, 2.70, 2.79],
    'Sales_Volume_Forecast': [43.61, 46.62, 47.39, 49.53, 51.44],
    'Total_Production_Value': [440.7, 727.3, 1554, 1508, 2793],
    'Total_Export_Value': [154.1, 71.01, 88.39, 178.96, 286.92]
}
df = pd.DataFrame(data)

X = df[['Sales_Forecast', 'Average_Price_Forecast', 'Sales_Volume_Forecast']]
y_production = df['Total_Production_Value']
y_export = df['Total_Export_Value']

X_train, X_test, y_production_train, y_production_test = train_test_split(X, y_production, test_size=0.2, random_state=42)
X_train_export, X_test_export, y_export_train, y_export_test = train_test_split(X, y_export, test_size=0.2, random_state=42)

model_production = LinearRegression()
model_production.fit(X_train, y_production_train)

model_export = LinearRegression()
model_export.fit(X_train_export, y_export_train)

future_data = np.array([[142.71, 2.94, 53.53], [150.96,3.08,55.49], [158.9,3.21,57.45]])  # 2024, 2025, 2026

future_production_values = model_production.predict(future_data)
future_export_values = model_export.predict(future_data)

df = pd.DataFrame(data)

X = df[['Sales_Forecast', 'Average_Price_Forecast', 'Sales_Volume_Forecast']]
y = df['Total_Production_Value']

model = LinearRegression()
model.fit(X, y)

future_data = np.array([
    [142.71, 2.94, 53.53*1.1],  # 2024
    [150.96, 3.08, 55.49*1.2],  # 2025
    [158.9, 3.21, 57.45*1.5]    # 2026
])

future_production_values = model.predict(future_data)

years_future = [2024, 2025, 2026]

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(df['Year'], df['Total_Production_Value'], marker='o', label='Actual Production Value', color='blue')
plt.plot(years_future, future_production_values, marker='x', label='Predicted Production Value', linestyle='--',
         color='red')
plt.xlabel('Year')
plt.ylabel('Total Production Value')
plt.title('Total Production Value\'s Prediction')
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)
plt.plot(df['Year'], df['Total_Export_Value'], marker='o', label='Actual Export Value', color='blue')
plt.plot(years_future, future_export_values, marker='x', label='Predicted Export Value', linestyle='--', color='red')
plt.xlabel('Year')
plt.ylabel('Total Export Value')
plt.title('Total Export Value\'s Prediction')
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()
