import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression

data = {
    'Year': [2018, 2019, 2020, 2021, 2022, 2023],
    'Total_sales': [75.00, 91.10, 93.90, 102.60, 123.60, 133.90],
    'Average_price': [2.15, 2.26, 2.40, 2.55, 2.70, 2.79],
    'Sales_volume': [41.43, 43.61, 46.62, 47.39, 49.53, 51.44]
}

df = pd.DataFrame(data)

df.set_index('Year', inplace=True)
print(df)

df['Total_sales'].plot(title='Pet Food Sales', figsize=(10, 6))
plt.xlabel('Year')
plt.ylabel('Total_sales')
plt.show()
result = adfuller(df['Total_sales'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])
df['Sales_diff'] = df['Total_sales'].diff().dropna()
result_diff = adfuller(df['Sales_diff'].dropna())
print('ADF Statistic after differencing:', result_diff[0])
print('p-value after differencing:', result_diff[1])
print('Critical Values after differencing:', result_diff[4])

df['Sales_diff'].plot(title='Differenced Sales', figsize=(10, 6))
plt.xlabel('Year')
plt.ylabel('Sales Difference')
plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plot_acf(df['Sales_diff'].dropna(), ax=ax1)
plot_pacf(df['Sales_diff'].dropna(), ax=ax2)
plt.show()

model = ARIMA(df['Total_sales'], order=(2, 1, 2))
model_fit = model.fit()
print(model_fit.summary())
forecast_steps = 3
forecast = model_fit.forecast(steps=forecast_steps)
forecast_index = pd.Index(range(2024, 2024 + forecast_steps), name='Year')

print(f'Forecast results for the next three years (in billions of dollars):')
for year, forecast_value in zip(forecast_index, forecast):
    print(f'{year}: {forecast_value:.2f} Billion $')
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Total_sales'], label='Historical data', marker='o')
plt.plot(forecast_index, forecast, label='Forecast data', color='red', marker='x')
plt.title('Pet Food Sales Prediction')
plt.xlabel('Year')
plt.ylabel('Total_sales')
plt.legend()
plt.grid(True)
plt.show()
df = pd.DataFrame(data)
df.set_index('Year', inplace=True)
normal_columns = ['Average_price', 'Sales_volume']
X = df.index.values.reshape(-1, 1)
years_to_predict = np.arange(2024, 2027).reshape(-1, 1)

predictions = {}

for column in normal_columns:
    y = df[[column]]
    model = LinearRegression().fit(X, y)
    predicted_values = model.predict(years_to_predict)
    predictions[column] = predicted_values
for column, predicted_vals in predictions.items():
    print(f"Predicted {column} for years 2024-2026:")
    for year, val in zip(np.arange(2024, 2027), predicted_vals.flatten()):
        print(f"Year {year}: {val:.2f} ")
plt.figure(figsize=(10, 8))
for i, column in enumerate(normal_columns, 1):
    plt.subplot(len(normal_columns), 1, i)
    plt.scatter(df.index, df[column], label='Actual_data', color='blue')
    plt.plot(np.arange(2024, 2027), predictions[column], label='Predicted_data', color='red',marker='x')
    plt.xlabel('Year')
    plt.ylabel(column)
    plt.title(f'{column} Linear fitting and prediction')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
