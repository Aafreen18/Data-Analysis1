import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------
# 1. Load dataset from CSV
# -----------------------------------
# Replace filename if needed
df = pd.read_csv("dataset1.csv")

# -----------------------------------
# 2. Change attribute type (Year)
# -----------------------------------
# Convert "2000-01" → 2000
df["Year_start"] = df["Year"].str[:4].astype(int)

# -----------------------------------
# 3. Check NULL values
# -----------------------------------
print("\nTotal Null values in each column:\n")
print(df.isna().sum())

# -----------------------------------
# 4. Remove rows with NULL values
# -----------------------------------
df_clean = df.dropna()
print(f"\nDataframe after removing NULLs: {df_clean}")

# -----------------------------------
# 5. Pearson correlation matrix
# -----------------------------------
corr_matrix = df_clean.select_dtypes(include=np.number).corr()

print("\nCorrelation Matrix:\n")
print(corr_matrix)

# -----------------------------------
# 6. Correlation heatmap
# -----------------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Coal Attributes")
plt.tight_layout()
plt.show()
