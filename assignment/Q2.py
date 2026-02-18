import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

current_script_path = Path(__file__).resolve().parent
data_file_path = current_script_path.parent / "data" / "USA_cars_datasets.csv"

# 1. Load and inspect the data
data = pd.read_csv(data_file_path)

# Keep only relevant variables
cars_data = data[["price", "year", "mileage"]]

# Check dimensions and missing values
print("Data dimensions:", cars_data.shape)
print("\nMissing values:")
print(cars_data.isna().sum())
print("\nHead of the data:")
print(cars_data.head())

# 2. Max-min normalize year and mileage
normalized_data = cars_data.copy()

for column_name in ["year", "mileage"]:
    normalized_data[column_name] = (
        normalized_data[column_name] - normalized_data[column_name].min()
    ) / (normalized_data[column_name].max() - normalized_data[column_name].min())

# 3. Train-test split (80% train, 20% test)
feature_data = normalized_data[["year", "mileage"]]
target_price = normalized_data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    feature_data,
    target_price,
    test_size=0.20,
    random_state=42
)

# 4. kNN regression for multiple k values
k_values = [3, 10, 25, 50, 100, 300]
mean_squared_errors = {}

for k in k_values:
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    predicted_prices = knn_model.predict(X_test)

    mse = mean_squared_error(y_test, predicted_prices)
    mean_squared_errors[k] = mse

    # Scatterplot: actual vs predicted prices
    plt.figure()
    plt.scatter(y_test, predicted_prices)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"kNN Regression: k = {k}")
    plt.show()

# 5. Display MSE results and identify optimal k
print("\nMean Squared Error by k:")
for k, mse in mean_squared_errors.items():
    print(f"k = {k}: MSE = {mse:,.2f}")

optimal_k = min(mean_squared_errors, key=mean_squared_errors.get)
print(f"\nOptimal k based on test MSE: {optimal_k}")
