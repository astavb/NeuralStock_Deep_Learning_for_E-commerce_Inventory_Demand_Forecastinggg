import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

# Data path
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ecommerce_inventory_demand.csv')

# Load data
df = pd.read_csv(DATA_PATH)

# Display basic info
print("Data shape:", df.shape)
print("\nData info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

# Data preprocessing
# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Assuming 'unit_price' represents sales amount per transaction
# If units_sold is present and not null, we could compute total_sales = units_sold * unit_price
# But since units_sold seems to have missing values, we'll use unit_price as sales
df.rename(columns={'unit_price': 'sales'}, inplace=True)

# Handle missing values in sales
df['sales'] = df['sales'].fillna(df['sales'].median())

# Aggregate sales by date for time series analysis
daily_sales = df.groupby('date')['sales'].sum().reset_index()
daily_sales.set_index('date', inplace=True)
daily_sales = daily_sales.asfreq('D').fillna(0)  # Fill missing dates with 0

print("\nDaily sales summary:")
print(daily_sales.describe())

# 1. Visualize sales distributions
plt.figure(figsize=(15, 10))

# Histogram of sales
plt.subplot(2, 3, 1)
sns.histplot(df['sales'], bins=50, kde=True)
plt.title('Distribution of Sales Amounts')
plt.xlabel('Sales Amount')
plt.ylabel('Frequency')

# Boxplot of sales
plt.subplot(2, 3, 2)
sns.boxplot(y=df['sales'])
plt.title('Boxplot of Sales Amounts')
plt.ylabel('Sales Amount')

# Daily sales over time
plt.subplot(2, 3, 3)
daily_sales['sales'].plot()
plt.title('Daily Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')

# Sales by product category
plt.subplot(2, 3, 4)
category_sales = df.groupby('product_category')['sales'].sum().sort_values(ascending=False)
category_sales.plot(kind='bar')
plt.title('Total Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)

# Sales by day of week
plt.subplot(2, 3, 5)
day_sales = df.groupby('day_of_week')['sales'].mean().sort_index()
day_sales.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_sales.plot(kind='bar')
plt.title('Average Sales by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Average Sales')
plt.xticks(rotation=45)

# Sales by month
plt.subplot(2, 3, 6)
month_sales = df.groupby('month')['sales'].mean().sort_index()
month_sales.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_sales.plot(kind='bar')
plt.title('Average Sales by Month')
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 2. Detect seasonality using decomposition plots
# Perform seasonal decomposition
decomposition = seasonal_decompose(daily_sales['sales'], model='additive', period=7)  # Weekly seasonality

plt.figure(figsize=(15, 10))

plt.subplot(4, 1, 1)
plt.plot(daily_sales['sales'])
plt.title('Original Time Series')
plt.ylabel('Sales')

plt.subplot(4, 1, 2)
plt.plot(decomposition.trend)
plt.title('Trend Component')
plt.ylabel('Trend')

plt.subplot(4, 1, 3)
plt.plot(decomposition.seasonal)
plt.title('Seasonal Component')
plt.ylabel('Seasonal')

plt.subplot(4, 1, 4)
plt.plot(decomposition.resid)
plt.title('Residual Component')
plt.ylabel('Residual')

plt.tight_layout()
plt.show()

# 3. Compute autocorrelation (ACF/PACF)
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plot_acf(daily_sales['sales'], lags=30, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')

plt.subplot(1, 2, 2)
plot_pacf(daily_sales['sales'], lags=30, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()

# 4. Profile missing values and outliers
# Missing values profile
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0]
if len(missing_data) > 0:
    missing_data.plot(kind='bar')
    plt.title('Missing Values by Column')
    plt.xlabel('Column')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=45)
else:
    plt.text(0.5, 0.5, 'No missing values found', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Missing Values Check')

# Outliers detection using IQR method
plt.subplot(1, 2, 2)
Q1 = df['sales'].quantile(0.25)
Q3 = df['sales'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['sales'] < lower_bound) | (df['sales'] > upper_bound)]
non_outliers = df[(df['sales'] >= lower_bound) & (df['sales'] <= upper_bound)]

plt.scatter(range(len(non_outliers)), non_outliers['sales'], alpha=0.5, label='Normal')
plt.scatter(range(len(non_outliers), len(non_outliers) + len(outliers)), outliers['sales'], color='red', alpha=0.5, label='Outliers')
plt.axhline(y=upper_bound, color='r', linestyle='--', label='Upper Bound')
plt.axhline(y=lower_bound, color='r', linestyle='--', label='Lower Bound')
plt.title('Outliers Detection in Sales')
plt.xlabel('Index')
plt.ylabel('Sales Amount')
plt.legend()

plt.tight_layout()
plt.show()

print(f"\nOutliers detected: {len(outliers)} out of {len(df)} records ({len(outliers)/len(df)*100:.2f}%)")
print(f"Lower bound: {lower_bound:.2f}")
print(f"Upper bound: {upper_bound:.2f}")

# Additional outlier analysis
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(x='product_category', y='sales', data=df)
plt.title('Sales Distribution by Product Category')
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
sns.boxplot(x='is_promotion', y='sales', data=df)
plt.title('Sales Distribution by Promotion Status')
plt.xticks([0, 1], ['No Promotion', 'Promotion'])

plt.subplot(1, 3, 3)
sns.scatterplot(x='stock_on_hand', y='sales', data=df, alpha=0.5)
plt.title('Sales vs Stock on Hand')

plt.tight_layout()
plt.show()