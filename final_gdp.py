import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Minimal, reusable functions to train/evaluate the GDP model exported from the notebook.

TARGET_COL = "GDPC1"
FEATURE_COLS = ["PCEPILFE_PC1", "GPDIC1", "GCEC1", "NETEXC"]

_POSSIBLE_COUNTRY_COLS = [
    "country", "Country", "COUNTRY", "LOCATION", "Country Name", "country_name"
]

def train_and_eval(csv_path, country=None):
    """Train a linear regression on the CSV and return useful artifacts.

    Returns a dict with keys:
      - model, scaler, feature_cols
      - X_test, y_test, years_test, y_pred
      - r2, rmse, coef_df
    """
    df = pd.read_csv(csv_path)

    # If a country filter is requested and a matching column exists, apply it
    country_col = None
    if country is not None:
        for c in _POSSIBLE_COUNTRY_COLS:
            if c in df.columns:
                country_col = c
                break
        if country_col is not None:
            df = df[df[country_col] == country].reset_index(drop=True)


    # Basic cleaning: coerce numeric and drop rows with missing required fields
    for col in [TARGET_COL] + FEATURE_COLS + ["observation_date"]:
        # observation_date may be non-numeric (dates) but keep as string for plotting
        if col != "observation_date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS + ["observation_date"]).reset_index(drop=True)

    split_idx = int(len(df) * 0.8)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    years = df["observation_date"].astype(str).values

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    years_test = years[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    coef_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Coefficient (per 1 std increase)": model.coef_
    })

    # Correlations between target and features on the filtered test set
    try:
        corr_df = df[[TARGET_COL] + FEATURE_COLS].corr()[TARGET_COL].drop(TARGET_COL)
    except Exception:
        corr_df = None

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "X_test": X_test,
        "y_test": y_test,
        "years_test": years_test,
        "y_pred": y_pred,
        "r2": r2,
        "rmse": rmse,
        "coef_df": coef_df,
        "correlations": corr_df,
        "country": country,
        "country_col": country_col,
    }

def predictions_dataframe(result):
    """Return a pandas DataFrame indexed by years_test with Actual and Predicted columns."""
    df = pd.DataFrame({
        "Actual": result["y_test"],
        "Predicted": result["y_pred"]
    }, index=pd.Index(result["years_test"], name="Year"))
    return df


def get_countries(csv_path):
    """Return list of unique countries found in the CSV using common country column names.

    If no country-like column is found, returns an empty list.
    """
    df = pd.read_csv(csv_path)
    for c in _POSSIBLE_COUNTRY_COLS:
        if c in df.columns:
            vals = sorted(df[c].dropna().unique().tolist())
            return vals
    return []
