import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from io import StringIO
from pandas.tseries.offsets import QuarterEnd

# Calculate quarters needed to reach 2030-Q4 from last data point
def calculate_quarters_to_2030(last_date):
    target_date = pd.Timestamp('2030-12-31')
    quarters_between = pd.date_range(start=last_date, end=target_date, freq='QE')
    return len(quarters_between)

# Your data
data = """Date,GDP,PCEPILFE_PC1,GCEC1,GPDIC1,NETEXP1
2022-01-01,21932.71,15098.34305,4272.382,3665.467,-1106.84
2022-04-01,21967.045,15222.98501,4176.99,3652.407,-1093.044
2022-07-01,22125.625,15295.07804,4105.292,3667.845,-953.091
2022-10-01,22278.345,15325.78527,4184.98,3705.121,-944.517
2023-01-01,22439.607,15495.18892,4108.542,3742.908,-912.191
2023-04-01,22580.499,15552.95782,4180.752,3773.544,-928.113
2023-07-01,22840.989,15671.12125,4275.527,3821.238,-926.169
2023-10-01,23033.78,15788.75487,4313.169,3865.061,-934.148
"""

# Load data
df = pd.read_csv(StringIO(data), parse_dates=['Date'])

# Calculate number of quarters needed to reach 2030-Q4
num_quarters = calculate_quarters_to_2030(df['Date'].iloc[-1])
last_quarter = pd.Period(df['Date'].iloc[-1], freq='Q').strftime('Q%q')
print(f"\nProjecting {num_quarters} quarters from {df['Date'].iloc[-1].strftime('%Y')}-{last_quarter} to 2030-Q4")

# Function to predict through 2030 using linear regression
def predict_through_2030(data, start_date, num_periods):
    # Create time index
    X = np.arange(len(data)).reshape(-1, 1)
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, data)
    
    # Predict quarters through 2030
    future_X = np.arange(len(data), len(data) + num_periods).reshape(-1, 1)
    predictions = model.predict(future_X)
    
    # Create future dates (quarter-ends) then convert to quarter-starts YYYY-01-01, YYYY-04-01, ...
    future_dates = pd.date_range(start=start_date, periods=num_periods, freq='QE')
    # Convert each quarter-end to its quarter-start (first day of the quarter)
    quarter_starts = [pd.Period(d, freq='QE').start_time for d in future_dates]

    return pd.Series(predictions, index=quarter_starts)

# Make predictions for each column
predictions = {}
for column in df.columns:
    if column != 'Date':
        predictions[column] = predict_through_2030(df[column].values, df['Date'].iloc[-1], num_quarters)

# Combine predictions into a DataFrame
future_df = pd.DataFrame(predictions)

# Print historical data for comparison
print("\n=== Historical Data (Last Year) ===")
print(df.tail(4).set_index('Date')[['GDP', 'PCEPILFE_PC1', 'GCEC1', 'GPDIC1', 'NETEXP1']].round(3))

print("\n=== Quarterly Projections through 2030 ===")
print(future_df.round(3))

# Save quarterly forecasts to CSV
quarterly_csv = "forecast_through_2030.csv"
future_df.round(3).to_csv(quarterly_csv, index_label='Date')

# Calculate annualized growth rates and save to CSV
annual_growth = {}
for column in future_df.columns:
    start_value = df[column].iloc[-1]
    end_value = future_df[column].iloc[-1]
    years = (pd.Timestamp('2030-12-31') - df['Date'].iloc[-1]).days / 365.25
    annual_growth[column] = ((end_value / start_value) ** (1 / years) - 1) * 100

growth_df = pd.Series(annual_growth, name='AvgAnnualGrowthPct')
growth_csv = "growth_rates_through_2030.csv"
growth_df.round(3).to_csv(growth_csv, header=True)

print("\nSaved quarterly forecast to:", quarterly_csv)
print("Saved annual growth rates to:", growth_csv)