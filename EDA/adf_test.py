#  Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf


# Load Your Data (Assuming you already have it loaded as 'data')

data = pd.read_csv('../Data/Augmented_Combined_data.csv', parse_dates=['Time'], index_col='Time')

# Split Original & Augmented
original_data = data[data['augmented'] == 0]
augmented_data = data[data['augmented'] != 0]

#  Identify Numeric Columns (excluding non-numeric ones)
exclude_cols = ['Location', 'augmented']
numeric_cols = [col for col in data.columns if col not in exclude_cols]

#  ADF Test for Stationarity
adf_orig = adfuller(original_data['Power'])
adf_aug = adfuller(augmented_data['Power'])

print(f"ADF Original - p-value: {adf_orig[1]:.5f}")
print(f"ADF Augmented - p-value: {adf_aug[1]:.5f}")


# Interpretation
if adf_orig[1] < 0.05:
    print("Original Power: Stationary")
else:
    print("Original Power: Non-stationary")

if adf_aug[1] < 0.05:
    print("Augmented Power: Stationary")
else:
    print("Augmented Power: Non-stationary")