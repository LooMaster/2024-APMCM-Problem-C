
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

data={
    'Year':[2019,2020,2021,2022,2023],
    'Cat_count':[4412,4862,5806,6536,6980],
    'Dog_count':[5503,5222,5429,5119,5175],
    'Per_capita_disposable_income':[30733,32189,35128,36883,39218],
    'Urbanization_rate':[44.38,45.4,46.7,47.7,48.3],
    'Total_population':[1409.67,1411.75,1412.6,1412.12,1410.08],
    'Cat_market_size':[798,884,1060,1231,1305],
    'Dog_market_size':[1210,1181,1430,1475,1488],
    'Number_of_hospital':[21000,23000,18000,27300,30000]
}

df=pd.DataFrame(data)

normal_columns = []

for column in df.columns:
    if column == 'Year':
        continue
    shapiro_test = stats.shapiro(df[column])
    print(f"{column} \'s Shapiro-Wilk Normality test results：")
    print(f"Test statistics: {shapiro_test[0]}, Probability values associated with test statistics (p-value): {shapiro_test[1]}\n")
    if shapiro_test[1] >= 0.05:
        normal_columns.append(column)
        print(f"{column} There was no significant deviation from the normal distribution（p >= 0.05）\n  ")
    else:
        print(f"{column} Significant deviation from normal distribution（p < 0.05）\n ")
if len(normal_columns) >= 2:
    df_normal = df[normal_columns]
    corr_matrix = df_normal.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
    plt.title('Pearson correlation coefficient_heatmap')
    plt.show()
else:
    print("There are not enough columns that do not deviate from the normal distribution to plot the heat zone.")

normal_columns = ['Cat_count', 'Dog_count']
X = df[['Year']]
years_to_predict = np.arange(2024, 2028).reshape(-1, 1)
predictions = {}
for column in normal_columns:
    y = df[[column]]
    model = LinearRegression().fit(X, y)
    predicted_values = model.predict(years_to_predict)
    predictions[column] = predicted_values.flatten()
plt.figure(figsize=(8,6))
for i, column in enumerate(normal_columns, 1):
    plt.subplot(len(normal_columns), 1, i)
    plt.scatter(df['Year'], df[column], label='Actual_data', color='blue')
    plt.plot(np.arange(2019, 2028), np.concatenate((df[[column]].values, predictions[column].reshape(-1, 1))), label='Predicted_data', color='red', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel(column)
    plt.title(f'{column} Linear fitting and prediction')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()

normal_columns = ['Per_capita_disposable_income', 'Urbanization_rate','Total_population']
X = df[['Year']]
years_to_predict = np.arange(2024, 2028).reshape(-1, 1)
predictions = {}
for column in normal_columns:
    y = df[[column]]
    model = LinearRegression().fit(X, y)
    predicted_values = model.predict(years_to_predict)
    predictions[column] = predicted_values.flatten()
plt.figure(figsize=(8,9))
for i, column in enumerate(normal_columns, 1):
    plt.subplot(len(normal_columns), 1, i)
    plt.scatter(df['Year'], df[column], label='Actual_data', color='blue')
    plt.plot(np.arange(2019, 2028), np.concatenate((df[[column]].values, predictions[column].reshape(-1, 1))), label='Predicted_data', color='red', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel(column)
    plt.title(f'{column} Linear fitting and prediction')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()

normal_columns = ['Cat_market_size', 'Dog_market_size','Number_of_hospital']
X = df[['Year']]
years_to_predict = np.arange(2024, 2028).reshape(-1, 1)
predictions = {}
for column in normal_columns:
    y = df[[column]]
    model = LinearRegression().fit(X, y)
    predicted_values = model.predict(years_to_predict)
    predictions[column] = predicted_values.flatten()

plt.figure(figsize=(8,9))
for i, column in enumerate(normal_columns, 1):
    plt.subplot(len(normal_columns), 1, i)
    plt.scatter(df['Year'], df[column], label='Actual_data', color='blue')
    plt.plot(np.arange(2019, 2028), np.concatenate((df[[column]].values, predictions[column].reshape(-1, 1))), label='Predicted_data', color='red', linestyle='--')
    plt.xlabel('Year')
    plt.ylabel(column)
    plt.title(f'{column} Linear fitting and prediction')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()


