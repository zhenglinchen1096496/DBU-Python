# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('covid_data.csv')

# 1. Data Exploration
# Display the first few rows of the dataset to get an overview
print("First 5 rows of the dataset:\n", df.head())

# Display summary statistics for numerical columns
print("\nSummary statistics:\n", df.describe())

# Check for missing values in the dataset
print("\nMissing values:\n", df.isnull().sum())


# 2. Data Visualization: Correlation heatmap
# Visualize the correlation between numerical variables using a heatmap
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()



# Visualization: Distribution of new cases by date
# This helps to understand how new cases changed over dates
plt.figure(figsize=(8, 6))
sns.boxplot(x='date', y='new_cases_smoothed', data=df)
plt.title('new_cases_smoothed by date')
plt.show()



# Visualization: Distribution of total death by countries
# Aggregate data to get the total deaths per million for each continent
continent_data = df.groupby('continent')['total_deaths_per_million'].sum().reset_index()

# Plot the aggregated data
plt.figure(figsize=(8, 6))
plt.bar(continent_data['continent'], continent_data['total_deaths_per_million'])
plt.title("Bar Plot of Total Deaths per Million by Continent")
plt.xlabel("Continent")
plt.ylabel("Total Deaths per Million")
plt.xticks(rotation=90)
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(df['date'], df['new_cases_smoothed'])  # Pass x and y as series, not as keyword arguments
plt.title("Line Plot of new cases")
plt.xlabel("Date")
plt.ylabel("New Cases Smoothed")
plt.show()


# Visualize Confusion Matrix using a heatmap
# plt.figure(figsize=(6, 4))
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# Ensure 'continent' and numeric columns have no NaNs and select a subset
# numeric_columns = ['total_deaths_per_million', 'total_cases_per_million']
# subset_df = df[numeric_columns + ['continent']].dropna()

# Pairplot with hue
# sns.pairplot(subset_df, hue='continent')
# plt.show()