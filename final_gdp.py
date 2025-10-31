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

    # capture last observed features and year to enable simple forecasting
    last_features = None
    last_year = None
    try:
        last_row = df.iloc[-1]
        last_features = last_row[FEATURE_COLS].astype(float).values
        # try to extract a 4-digit year from observation_date
        raw = last_row["observation_date"]
        try:
            last_year = int(str(raw)[:4])
        except Exception:
            try:
                last_year = pd.to_datetime(raw).year
            except Exception:
                last_year = None
    except Exception:
        last_features = None
        last_year = None

    # compute simple recent growth rates (mean pct change over last 3 observations) per feature
    growth_rates = None
    try:
        pct = df[FEATURE_COLS].pct_change().dropna()
        if len(pct) >= 1:
            growth_rates = pct.tail(3).mean().fillna(0).values
        else:
            growth_rates = np.zeros(len(FEATURE_COLS))
    except Exception:
        growth_rates = np.zeros(len(FEATURE_COLS))

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
        "last_features": last_features,
        "last_year": last_year,
        "growth_rates": growth_rates,
    }

def predictions_dataframe(result):
    """Return a pandas DataFrame indexed by years_test with Actual and Predicted columns."""
    df = pd.DataFrame({
        "Actual": result["y_test"],
        "Predicted": result["y_pred"]
    }, index=pd.Index(result["years_test"], name="Year"))
    return df


def forecast_next_years(result, n_years=5, method="trend"):
    """Forecast the target for the next n_years using the trained model.

    Parameters
    - result: dict returned by train_and_eval
    - n_years: how many future years to predict
    - method: 'trend' to project features using recent pct changes, 'constant' to hold last values

    Returns a DataFrame with index Year and column Predicted.
    """
    model = result.get("model")
    scaler = result.get("scaler")
    feature_cols = result.get("feature_cols")
    last_feats = result.get("last_features")
    last_year = result.get("last_year")
    growth_rates = result.get("growth_rates")

    if model is None or scaler is None or last_feats is None or last_year is None:
        raise ValueError("Result must contain trained model, scaler, last_features and last_year for forecasting.")

    preds = []
    years = []
    current_feats = last_feats.copy().astype(float)

    for i in range(1, n_years + 1):
        year = last_year + i
        years.append(year)

        if method == "trend" and growth_rates is not None:
            # apply growth rates multiplicatively each year
            current_feats = current_feats * (1.0 + growth_rates)
        else:
            # constant: do not change features (keep last observed)
            current_feats = current_feats

        X_scaled = scaler.transform(current_feats.reshape(1, -1))
        y_pred = model.predict(X_scaled)[0]
        preds.append(y_pred)

    return pd.DataFrame({"Predicted": preds}, index=pd.Index([str(y) for y in years], name="Year"))



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
