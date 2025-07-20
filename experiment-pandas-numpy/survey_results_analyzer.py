import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the survey results CSV
csv_path = 'survey_results_public.csv'
df = pd.read_csv(csv_path)

# Basic info
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('Missing values per column:')
print(df.isnull().sum())

# Drop columns with too many missing values (threshold: 50%)
threshold = len(df) * 0.5
df = df.dropna(thresh=threshold, axis=1)

# Drop rows with missing values in remaining columns
clean_df = df.dropna()

# Show basic statistics
print('\nBasic statistics:')
print(clean_df.describe(include='all'))

# Correlation heatmap for numeric columns
plt.figure(figsize=(12,8))
sns.heatmap(clean_df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Example: Predict Salary based on other features (if available)
if 'ConvertedCompYearly' in clean_df.columns:
    # Select numeric features
    features = clean_df.select_dtypes(include=np.number).drop('ConvertedCompYearly', axis=1)
    target = clean_df['ConvertedCompYearly']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('\nSalary Prediction:')
    print('R2 score:', r2_score(y_test, y_pred))
    print('MSE:', mean_squared_error(y_test, y_pred))
else:
    print('No salary column found for prediction.')

# Useful statistics: Top countries, most common languages, etc.
if 'Country' in clean_df.columns:
    print('\nTop 10 countries:')
    print(clean_df['Country'].value_counts().head(10))
if 'LanguageHaveWorkedWith' in clean_df.columns:
    print('\nMost common languages:')
    lang_series = clean_df['LanguageHaveWorkedWith'].str.split(';').explode()
    print(lang_series.value_counts().head(10))

# You can extend this script to analyze other columns as needed.
