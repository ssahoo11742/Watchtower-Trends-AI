import pandas as pd

# Read the original CSV
df = pd.read_csv('companies.csv')

# Keep only Ticker and Name columns
df_filtered = df[['Ticker', 'Name']]

# Save to a new CSV file
df_filtered.to_csv('companies_filtered.csv', index=False)

print(f"Original CSV: {len(df)} rows, {len(df.columns)} columns")
print(f"Filtered CSV: {len(df_filtered)} rows, {len(df_filtered.columns)} columns")
print(f"Columns kept: {list(df_filtered.columns)}")
print("\nSaved as 'companies_filtered.csv'")