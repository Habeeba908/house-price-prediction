"""
Task 3: House Price Prediction
Predict house prices with Linear Regression.
Note: Boston Housing was removed from scikit-learn (ethical concerns),
so we use the California Housing dataset - a standard replacement.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load the dataset
data = fetch_california_housing(as_frame=True)
df = data.frame  # target column: MedHouseVal (in $100,000s)
print("First 5 rows:\n", df.head(), "\n")
print("Shape:", df.shape)
print(df.describe(), "\n")

# Explore distributions
df.hist(figsize=(12, 8), bins=30, edgecolor="black")
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig("house_histograms.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(df.corr(), cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(df.columns)), df.columns, rotation=90)
plt.yticks(range(len(df.columns)), df.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("house_correlation.png", dpi=150)
plt.show()

# 2. Handle missing data
print("Missing values:\n", df.isnull().sum(), "\n")
df = df.fillna(df.median())  # no-op here, but safe practice

# 3. Split into train/test sets
X = df.drop(columns="MedHouseVal")
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Normalize inputs
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Linear Regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", round(mse, 4))
print("Root MSE (RMSE)         :", round(np.sqrt(mse), 4))
print("Mean Absolute Error     :", round(mean_absolute_error(y_test, y_pred), 4))
print("R2 Score                :", round(r2_score(y_test, y_pred), 4))

coef = pd.Series(model.coef_, index=X.columns).sort_values()
print("\nFeature coefficients:\n", coef)

# Actual vs predicted plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual Price ($100k)")
plt.ylabel("Predicted Price ($100k)")
plt.title("Actual vs Predicted House Prices")
plt.savefig("house_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()