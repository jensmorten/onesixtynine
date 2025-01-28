import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the data from a CSV file
filename = "pollofpolls.csv"
df = pd.read_csv(filename)

# Ensure the "Mnd" column is treated as a string (optional but recommended)
df["Mnd"] = df["Mnd"].astype(str)

# Add an index column for numerical representation of the months
df["Index"] = np.arange(len(df), 0, -1)

# Create an empty dictionary for the predictions
predictions = {"Mnd": "Februar-25"}

# Prepare the index for the new month to predict
new_index = [[len(df) + 1]]

# Loop through each column (except 'Mnd' and 'Index') to predict its value
for column in df.columns[1:-1]:  # Skip 'Mnd' and 'Index'
    # Separate features (Index) and target (current column)
    X = df[["Index"]]
    y = df[column]

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the value for "Februar-25"
    predictions[column] = model.predict(new_index)[0]

# Print the predicted values for "Februar-25"
print("Predicted values for 'Februar-25':")
for key, value in predictions.items():
    if key == "Mnd":
        print(f"{key}: {value}")
    else:
        print(f"{key}: {value:.2f}")
