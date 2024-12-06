import matplotlib.pyplot as plt
import pandas as pd

data = {
    'Year': [2024, 2025, 2026],
    '5% Tariff\'s market size (CNY)': [3136.125, 3353.425, 3570.725],
    '10% Tariff\'s market size (CNY)': [3205.95, 3423.25, 3640.55],
    '15% Tariff\'s market size (CNY)': [3275.775, 3493.075, 3710.375]
}

df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))

plt.plot(df['Year'], df['5% Tariff\'s market size (CNY)'], label='5% Tariff',color='green',alpha=0.7,linestyle='--',linewidth=2,marker='o')
plt.plot(df['Year'], df['10% Tariff\'s market size (CNY)'], label='10% Tariff',color='blue', alpha=0.6,linestyle='--',linewidth=2,marker='o')
plt.plot(df['Year'], df['15% Tariff\'s market size (CNY)'], label='15% Tariff', color='red',alpha=1,linestyle='--',linewidth=2,marker='o')

plt.title('The market size at different tariff levels changes over time')
plt.xlabel('Year')
plt.ylabel('Market size (CNY)')

plt.legend()

plt.grid(True)

plt.show()