import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------
# 1. Load the dataset
# ------------------------------------------------------------
def load_data(path):
    """
    Loads the CSV file from the given path.
    """
    df = pd.read_csv(path)
    return df


# ------------------------------------------------------------
# 2. Clean the data
# ------------------------------------------------------------
def clean_data(df):
    """
    Cleans the dataset:
    - Removes constant/useless columns
    - Handles missing values
    """

    # Drop constant columns
    constant_columns = ['EmployeeCount', 'StandardHours', 'Over18']
    df.drop(columns=[col for col in constant_columns if col in df.columns],
            inplace=True)

    # Handle missing numeric values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Handle missing categorical values
    df.fillna(df.mode().iloc[0], inplace=True)

    return df


# ------------------------------------------------------------
# 3. Encode categorical variables
# ------------------------------------------------------------
def encode_data(df):
    """
    Encodes categorical columns using LabelEncoder.
    """

    # Explicitly encode target variable
    if 'Attrition' in df.columns:
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    # Encode remaining categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df


# ------------------------------------------------------------
# 4. Run all steps
# ------------------------------------------------------------
if __name__ == "__main__":
    file_path = r"D:\Nivi\Python\EmployeeAttrition\Employee-Attrition - Employee-Attrition.csv"

    df = load_data(file_path)
    df = clean_data(df)
    df = encode_data(df)

    # Save cleaned data
    df.to_csv("cleaned_employee_attrition.csv", index=False)

    print("✅ Data cleaning and encoding completed successfully!")
