import pandas as pd
from statsmodels.tsa.api import VAR

# Load the data from a CSV file
filename = "pollofpolls.csv"
df = pd.read_csv(filename)

# Ensure the "Mnd" column is treated as a string
df["Mnd"] = df["Mnd"].astype(str)

# Drop the "Mnd" column for the VAR model (it only works with numerical data)
df_numeric = df.drop(columns=["Mnd"])

# Check if the data is stationary (optional, as VAR works better on stationary data)
# Perform differencing if needed (example commented below)
# df_numeric = df_numeric.diff().dropna()

# Fit the VAR model
model = VAR(df_numeric)
results = model.fit(maxlags=2)  # Adjust lags as needed

# Predict the next time step
forecast = results.forecast(df_numeric.values[-results.k_ar:], steps=1)

# Convert the forecast to a dictionary
predictions = dict(zip(df_numeric.columns, forecast[0]))

# Add the predicted row to the original DataFrame
predicted_month = "Februar-25"  # Name for the new row
df.loc[len(df)] = [predicted_month] + list(predictions.values())

# Print the forecasted values for "Februar-25"
print("Predicted values for 'Februar-25':")
for column, value in predictions.items():
    print(f"{column}: {value:.2f}")
