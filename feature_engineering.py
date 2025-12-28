import pandas as pd

def add_features(df):
    """
    Adds engineered features to improve attrition prediction.
    """

    # -------------------------------------------------
    # Ensure numeric type
    # -------------------------------------------------
    df['YearsAtCompany'] = pd.to_numeric(df['YearsAtCompany'], errors='coerce')
    df['MonthlyIncome'] = pd.to_numeric(df['MonthlyIncome'], errors='coerce')

    # Fill missing values
    df['YearsAtCompany'].fillna(0, inplace=True)
    df['MonthlyIncome'].fillna(0, inplace=True)

    # -------------------------------------------------
    # 1. Tenure Group
    # 0 = 0–2 years
    # 1 = 3–5 years
    # 2 = 6–10 years
    # 3 = 10+ years
    # -------------------------------------------------
    df['TenureGroup'] = pd.cut(
        df['YearsAtCompany'],
        bins=[-1, 2, 5, 10, 50],
        labels=[0, 1, 2, 3]
    )

    # -------------------------------------------------
    # 2. Income Group
    # 0 = Low
    # 1 = Medium
    # 2 = High
    # -------------------------------------------------
    df['IncomeGroup'] = pd.cut(
        df['MonthlyIncome'],
        bins=[-1, 3000, 7000, 100000],
        labels=[0, 1, 2]
    )

    # Convert category dtype to integer
    df['TenureGroup'] = df['TenureGroup'].astype(int)
    df['IncomeGroup'] = df['IncomeGroup'].astype(int)

    return df
