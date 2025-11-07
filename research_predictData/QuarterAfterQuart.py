import pandas as pd
from io import StringIO

# Paste your data as a multiline string
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

# Read the data into a DataFrame
df = pd.read_csv(StringIO(data), parse_dates=['Date'])

# Calculate percent change for each numeric column (except the Date)
percent_changes = df.iloc[:, 1:].pct_change() * 100

# Add suffix to make columns clear
percent_changes.columns = [col + '_PctChange' for col in percent_changes.columns]

# Combine original and percent change data
result = pd.concat([df, percent_changes], axis=1)

# Round for readability
result = result.round(3)



# Get the latest values (last row of original numeric columns)
latest_values = df.iloc[-1, 1:]  # Skip the Date column

# Get the latest percent changes (last row of percent changes)
latest_pct_changes = percent_changes.iloc[-1]

# Create a dictionary to store all future values
future_values = {}
num_steps = 28  # Total number of steps to project

# Initialize with current values
for col in df.columns[1:]:  # Skip the Date column
    future_values[col] = [latest_values[col]]
    pct_change = latest_pct_changes[f"{col}_PctChange"] / 100
    
    # Calculate future values for all steps
    for step in range(num_steps):
        next_value = future_values[col][-1] * (1 + pct_change)
        future_values[col].append(next_value)

# Create a DataFrame for all steps
columns = ['Current']
columns.extend([f'Step {i+1}' for i in range(num_steps)])

summary = pd.DataFrame(index=df.columns[1:])  # Skip Date column
for i, col_name in enumerate(columns):
    summary[col_name] = [future_values[col][i] for col in df.columns[1:]]

# Add percent change information
summary['Percent Change (%)'] = latest_pct_changes[df.columns[1:].map(lambda x: x + '_PctChange')]

# Round the summary for display
summary = summary.round(3)

# Display results vertically for each metric
print("\nDetailed Future Value Projections:")
print("================================")

for metric in df.columns[1:]:  # Skip Date column
    print(f"\n{metric}")
    print("-" * len(metric))
    print(f"Per-step change: {latest_pct_changes[metric + '_PctChange']:.3f}%")
    print("Progression:")
    for step, value in enumerate(future_values[metric]):
        if step == 0:
            print(f"Current:  {value:,.3f}")
        else:
            print(f"Step {step:2d}:  {value:,.3f}")
    
    # Calculate total change
    total_pct_change = (future_values[metric][-1] / future_values[metric][0] - 1) * 100
    print(f"Total Change: {total_pct_change:.3f}%")
    print()

# Also save the summary and the step-by-step series to CSV files
summary_csv = 'QuarterAfterQuart_summary.csv'
try:
    summary.to_csv(summary_csv, index_label='Metric')
except PermissionError:
    # If write is denied, try to write to a timestamped fallback filename
    import datetime, os, stat
    fallback = f"QuarterAfterQuart_summary_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
    try:
        if os.path.exists(summary_csv):
            # attempt to make it writable then remove it
            os.chmod(summary_csv, stat.S_IWRITE)
            os.remove(summary_csv)
    except Exception:
        pass
    summary.to_csv(fallback, index_label='Metric')
    summary_csv = fallback

# Create a step-indexed DataFrame (rows: Current, Step 1..Step N; cols: metrics)
step_index = ['Current'] + [f'Step {i+1}' for i in range(num_steps)]
steps_df = pd.DataFrame({metric: future_values[metric] for metric in df.columns[1:]}, index=step_index)
steps_csv = 'QuarterAfterQuart_steps.csv'
try:
    steps_df.round(3).to_csv(steps_csv, index_label='Step')
except PermissionError as e:
    # Attempt to clear read-only attribute and remove file, otherwise write to timestamped fallback
    import datetime, os, stat
    try:
        if os.path.exists(steps_csv):
            os.chmod(steps_csv, stat.S_IWRITE)
            os.remove(steps_csv)
            # retry write
            steps_df.round(3).to_csv(steps_csv, index_label='Step')
    except Exception:
        fallback_steps = f"QuarterAfterQuart_steps_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
        steps_df.round(3).to_csv(fallback_steps, index_label='Step')
        steps_csv = fallback_steps

print(f"\nSaved summary CSV to: {summary_csv}")
print(f"Saved per-step CSV to: {steps_csv}")
