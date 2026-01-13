import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ----------------------------------
# Load merged dataset
# ----------------------------------
df = pd.read_csv("merged_coalDataset.csv")

# ----------------------------------
# Clean Year column (extract numeric year)
# ----------------------------------
df["Year_num"] = df["Year"].str.extract(r"(\d{4})").astype(int)

# ----------------------------------
# Select required columns & drop NA
# ----------------------------------
df_lr = df[["Year_num", "Coal - Total"]].dropna()

X = df_lr[["Year_num"]]   # must be 2D
y = df_lr["Coal - Total"]

# ----------------------------------
# Train-test split
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------
# Train Linear Regression model
# ----------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------------
# Predictions
# ----------------------------------
y_pred = model.predict(X_test)

# ----------------------------------
# Evaluation
# ----------------------------------
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# ----------------------------------
# Visualization
# ----------------------------------
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Year")
plt.ylabel("Total Coal Production")
plt.title("Year vs Total Coal Production")
plt.show()
