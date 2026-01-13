import pandas as pd

df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")

df2.rename(columns={
    "Year-wise": "Year",
    "Coking Coal - Qty": "Coal - Coking",
    "Non-Coking Coal - Qty": "Coal - Non-coking",
    "Total Coal - Qty": "Coal - Total"
}, inplace=True)

merged_df = pd.merge(
    df1,
    df2,
    on=[
        "Year",
        "Coal - Coking",
        "Coal - Non-coking",
        "Coal - Total"
    ],
    how="outer"
)

merged_df.sort_values("Year", inplace=True)

print(merged_df)

merged_df.to_csv("merged_coalDataset.csv", index=False)
