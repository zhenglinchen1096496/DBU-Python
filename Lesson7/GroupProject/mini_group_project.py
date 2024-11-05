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
df = pd.read_csv('covid_data_total.csv')

# 1. Data Exploration
# Display the first few rows of the dataset to get an overview
print("First 5 rows of the dataset:\n", df.head())

# Display summary statistics for numerical columns
print("\nSummary statistics:\n", df.describe())

# Check for missing values in the dataset
print("\nMissing values:\n", df.isnull().sum())

# Convert 'date' into datetime type
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')




# 2. Visualize of heatmap: the correlation between numerical variables
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()



# 3. Visualization of box plot: Distribution of total cases by continent
# This helps to understand how new cases changed over dates
# plt.figure(figsize=(8, 6))
# sns.boxplot(x='continent', y='total_deaths', data=df)
# plt.title('total_deaths by continent')
# plt.show()


# # 4. Visualization of bar plot: Distribution of total death by continent
# # Aggregate data to get the total deaths per million for each continent
# continent_data = df.groupby('continent')['total_deaths'].sum().reset_index()
# plt.figure(figsize=(8, 6))
# plt.bar(continent_data['continent'], continent_data['total_deaths'])
# plt.title("Bar Plot of Total Deaths by Continent")
# plt.xlabel("Continent")
# plt.ylabel("Total Deaths")
# plt.xticks(rotation=90)
# plt.show()

# 5. Visualization of Line plot: Distribution of new cases by date
# plt.figure(figsize=(8, 6))
# plt.plot(df['date'], df['new_cases_smoothed']) # Pass x and y as series, not as keyword arguments
# plt.title("Line Plot of new cases by date")
# plt.xlabel("Date")
# plt.ylabel("New Cases Smoothed")
# plt.show()


# plt.figure(figsize=(8, 6))
# plt.plot(df['total_cases'], df['total_vaccinations'])
# plt.title("Line Plot of relationship between total cases and total_vaccinations")
# plt.xlabel("total_cases")
# plt.ylabel("total_vaccinations")
# plt.show()

# Select relevant numerical columns and the categorical 'continent' column for hue
# covid_data = df[['total_cases', 'new_cases_smoothed', 'total_deaths',  'total_vaccinations', 'continent']].dropna()
# Create the pair plot with 'continent' as the hue
# sns.pairplot(covid_data, hue='continent', height=2.5)
# plt.show()


# # Step 1: Define the 'high_impact' column based on a threshold for total cases
# threshold = 100000  # example threshold for high impact
# df['high_impact'] = (df['total_cases'] > threshold).astype(int)

# # Step 2: Select features and target variable
# X = df[['total_deaths', 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']]
# y = df['high_impact']

# # Step 3: Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 4: Train the Logistic Regression model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # Step 5: Make predictions on the test set
# predictions = model.predict(X_test)

# # Step 6: Evaluate the model
# accuracy = accuracy_score(y_test, predictions)
# report = classification_report(y_test, predictions)

# print("Accuracy:", accuracy)
# print("Classification Report:\n", report)

# # Step 7: Plot the confusion matrix
# conf_matrix = confusion_matrix(y_test, predictions)
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Impact', 'High Impact'], yticklabels=['Low Impact', 'High Impact'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix for COVID High Impact Classification')
# plt.show()


from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
df = pd.read_csv('covid_data_total.csv')

# Step 1: Choose the target variable and features
# Let's predict total_cases based on total_vaccinations and total_deaths
X = df[['total_vaccinations', 'total_deaths']]
y = df['total_cases']

# Step 2: Handle missing values (if any)
X.fillna(0, inplace=True)  # Replace NaN values with 0 or use appropriate imputation
y.fillna(0, inplace=True)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
