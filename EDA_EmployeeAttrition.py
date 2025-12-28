import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_data, clean_data, encode_data
from feature_engineering import add_features
from sklearn.ensemble import RandomForestClassifier

# ---------------------
# Load & preprocess data
# ---------------------
df = load_data(r"D:\Nivi\Python\EmployeeAttrition\Employee-Attrition - Employee-Attrition.csv")
df = clean_data(df)
df = encode_data(df)
df = add_features(df)

sns.set(style="whitegrid")

# ---------------------
# 1. Basic Info
# ---------------------
df.info()
print(df.describe())

# ---------------------
# 2. Target Distribution
# ---------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Attrition', data=df)
plt.title("Attrition Distribution (0 = Stay, 1 = Leave)")
plt.show()

# ---------------------
# 3. Selected Numerical Distributions
# ---------------------
num_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'TotalWorkingYears']

df[num_cols].hist(bins=20, figsize=(12,6))
plt.suptitle("Distribution of Key Numerical Features")
plt.show()

# ---------------------
# 4. Correlation Heatmap (Clean)
# ---------------------
plt.figure(figsize=(14,8))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# ---------------------
# 5. Attrition vs Key Features
# ---------------------
features = ['Age', 'MonthlyIncome', 'JobSatisfaction',
            'YearsAtCompany', 'OverTime', 'JobLevel']

for col in features:
    plt.figure(figsize=(6,4))
    if col in ['OverTime', 'JobLevel', 'JobSatisfaction']:
        sns.countplot(x=col, hue='Attrition', data=df)
    else:
        sns.boxplot(x='Attrition', y=col, data=df)
    plt.title(f"Attrition vs {col}")
    plt.show()

# ---------------------
# 6. Exploratory Feature Importance (Random Forest)
# ---------------------
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X, y)

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature',
            data=feature_importances.head(10))
plt.title("Top 10 Features Influencing Attrition")
plt.show()

print("Top 10 Features Driving Attrition:\n")
print(feature_importances.head(10))
