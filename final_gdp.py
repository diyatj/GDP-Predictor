import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Minimal, reusable functions to train/evaluate the GDP model exported from the notebook.

TARGET_COL = "GDPC1"
FEATURE_COLS = ["Real PCE", "GPDIC1", "GCEC1", "NETEXC"]

_POSSIBLE_COUNTRY_COLS = [
    "country", "Country", "COUNTRY", "LOCATION", "Country Name", "country_name"
]

#Train a linear regression on the CSV and return useful artifacts.
def train_and_eval(csv_path, country=None, actual_df=None):

    #makes dataframe from predicted csv for train/test
    df = pd.read_csv(csv_path)

    #make dataframe for actual csv
    if actual_df is not None:
        actual_df = actual_df.copy()
    else:
        actual_df = pd.read_csv(csv_path)

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
    for col in df.columns:
        if col != "observation_date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS + ["observation_date"]).reset_index(drop=True)

    # capture last observed features and year to enable simple forecasting
    # It extract a 4-digit year and quarter from observation_date
    last_features = None
    last_year = None
    last_quarter = None
    try:
        last_row = df.iloc[-1]
        last_features = last_row[FEATURE_COLS].astype(float).values
        # try to extract a 4-digit year and quarter from observation_date
        raw = last_row["observation_date"]
        try:
            last_date = pd.to_datetime(raw)
            last_year = last_date.year
            # Extract quarter: Q1=1, Q2=2, Q3=3, Q4=4
            last_quarter = (last_date.month - 1) // 3 + 1
        except Exception:
            try:
                last_year = int(str(raw)[:4])
                last_quarter = None
            except Exception:
                last_year = None
                last_quarter = None
    except Exception:
        last_features = None
        last_year = None
        last_quarter = None

    # compute simple recent growth rates (mean pct change over last 3 observations) per feature
    growth_rates = None

    # compute percent change
    try:
        pct = df[FEATURE_COLS].pct_change().dropna()
        if len(pct) >= 1:
            N = min(20, len(pct))
            growth_rates = pct.tail(N).mean().fillna(0).values
        else:
            growth_rates = np.zeros(len(FEATURE_COLS))
    except Exception:
        growth_rates = np.zeros(len(FEATURE_COLS))

    # Split into train/test (80/20 time-based split)
    split_idx = int(len(df) * 0.8)

    # Prepare feature matrix and target vector
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    years = df["observation_date"].astype(str).values

    # Split into train and test sets
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    years_test = years[split_idx:]

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # RMSE as percent of average actual GDP
    if len(y_test) > 0:
        avg_actual = np.mean(y_test)
        if avg_actual != 0:
            rmse_pct_mean = (rmse / avg_actual) * 100.0
        else:
            rmse_pct_mean = np.nan
    else:
        avg_actual = np.nan
        rmse_pct_mean = np.nan

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
        "coef_df": coef_df,
        "correlations": corr_df,
        "country": country,
        "country_col": country_col,
        "last_features": last_features,
        "last_year": last_year,
        "last_quarter": last_quarter,
        "growth_rates": growth_rates,
    }

# Create a DataFrame of predictions vs actuals
def predictions_dataframe(result):
    """Return a pandas DataFrame indexed by years_test with Actual and Predicted columns."""
    df = pd.DataFrame({
        "Predicted": result["y_pred"]
    }, index=pd.Index(result["years_test"], name="Year"))
    return df

# Forecast the next n_quarters using the trained model
def forecast_next_quarters(result, n_quarters=5, return_dates=True, shock_quarter_index=0):
    
    # Labeling the method for feature evolution
    model = result.get("model")
    scaler = result.get("scaler")
    feature_cols = result.get("feature_cols")
    last_feats = result.get("last_features")
    last_year = result.get("last_year")
    last_quarter = result.get("last_quarter", None)
    growth_rates = result.get("growth_rates")
    X_test = result.get("X_test")
    y_test = result.get("y_test")

    # Validate required components
    if model is None or scaler is None or last_feats is None or last_year is None:
        raise ValueError("Result must contain trained model, scaler, last_features and last_year for forecasting.")

    # Generate future features by applying growth rates
    if y_test is not None and len(y_test) > 1:
        t = np.arange(len(y_test))
        y_pred = result["y_pred"]

        m, b = np.polyfit(t, y_pred, 1)
    else:
        m, b = 0.0, result["y_pred"][-1]

    # Generate predictions for future quarters
    preds = []
    quarters = []
    for i in range(1, n_quarters + 1):
        t_future = len(y_test) - 1 + i
        y_future = m * t_future + b
        preds.append(y_future)

        quarter = last_year + (i / 4.0)
        quarters.append(quarter)


    # Format quarters as dates: 2026-01-01, 2026-04-01, 2026-07-01, 2026-10-01
    quarter_dates = []
    
    # If we have last_quarter info, start from the next quarter
    if last_quarter is not None:
        current_year = last_year
        current_quarter = last_quarter + 1
        if current_quarter > 4:
            current_quarter = 1
            current_year += 1
    else:
        # Fallback to old behavior if quarter info not available
        current_year = last_year
        current_quarter = 1
    
    for i in range(n_quarters):
        # Map quarter number to month: Q1->01, Q2->04, Q3->07, Q4->10
        month = (current_quarter - 1) * 3 + 1
        date_str = f"{current_year}-{month:02d}-01"
        quarter_dates.append(date_str)
        
        # Move to next quarter
        current_quarter += 1
        if current_quarter > 4:
            current_quarter = 1
            current_year += 1
    
    quarter_dates = pd.to_datetime(quarter_dates)

    #return pd.DataFrame({"Predicted": preds}, index=pd.Index(quarter_dates, name="Date"))
    return pd.DataFrame({"Forecast": np.array(preds, dtype=float)}, index=pd.Index(quarter_dates, name="observation_date"))

# Get list of unique countries from CSV
def get_countries(csv_path):
    df = pd.read_csv(csv_path)
    for c in _POSSIBLE_COUNTRY_COLS:
        if c in df.columns:
            vals = sorted(df[c].dropna().unique().tolist())
            return vals
    return []

# Load events/Shock Factors from a country-specific CSV
def load_events(events_csv_path):
    try:
        df = pd.read_csv(events_csv_path)
        return df
    except Exception:
        return pd.DataFrame(columns=["event", "severity", "gdp_impact", "growth_shock", "length", "recovery"])

# Apply a single event shock to the predictions
def apply_event_shock(preds, event_row, shock_year_index):

    # Keep original predictions for reference
    original_preds = preds.copy()
    preds = np.array(preds, dtype=float).copy()
    n_quarters = len(preds)
    if shock_year_index >= n_quarters:
        return preds
    
    #parameters
    gdp_start = preds[shock_year_index]
    gdp_impact_mult = float(event_row.get("gdp_impact", 1.0))
    growth_shock = float(event_row.get("growth_shock", 0.0)) / 100.0
    length = int(event_row.get("length", 0))
    recovery = int(event_row.get("recovery", 0))
    
    shock_values = []
    current_gdp = gdp_start

    # Generate shock path
    for i in range(length):
        current_gdp *= (1 + growth_shock)
        shock_values.append(current_gdp)
        
    if length == 0:
        shock_values = [gdp_start * gdp_impact_mult]
    else:
        # Rescale the shock values so that the last one matches gdp_mult
        final_target = gdp_start * gdp_impact_mult
        actual_final = shock_values[-1] if shock_values else gdp_start

        scale_factor = final_target / actual_final if actual_final != 0 else 1.0
        shock_values = [v * scale_factor for v in shock_values]

    # Apply the scaled shock path to predictions
    for i, gdp_val in enumerate(shock_values):
        idx = shock_year_index + i
        if idx < n_quarters:
            preds[idx] = gdp_val
            
    if recovery > 0:
        recovery_start = shock_year_index + length
        
        # Ensure recovery_start is within bounds
        if recovery_start >= n_quarters:
            # Event extends beyond forecast horizon; no room for recovery
            pass
        else:
            recovery_end_idx = min(shock_year_index + length + recovery, n_quarters - 1)
            recovery_target = original_preds[recovery_end_idx]
            
            # Use the last valid shock value as recovery starting point
            recovery_start_value = preds[recovery_start - 1] if recovery_start > 0 else preds[0]
            
            recovery_path = np.linspace(recovery_start_value, recovery_target, recovery + 1)[1:]
            for i, gdp_val in enumerate(recovery_path):
                idx = recovery_start + i
                if idx < n_quarters:
                    preds[idx] = gdp_val
    
    for idx in range(shock_year_index + length + recovery, n_quarters):
        preds[idx] = original_preds[idx]
    
    return preds
