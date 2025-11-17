import pandas as pd

# Load the CSV
df = pd.read_csv("companies_extended.csv")

# Remove the Description column
df = df.drop("Description", axis=1)

# Save back to CSV
df.to_csv("companies_extended.csv", index=False)

print("Description column removed successfully!")