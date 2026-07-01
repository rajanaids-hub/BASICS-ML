"""
Linear Regression – Basics of Machine Learning
=================================================
Predicts a continuous output value from one or more input features.

Example: Predict house prices based on size (square footage).
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def generate_data(n_samples: int = 100, noise: float = 15.0, seed: int = 42):
    """Generate a simple house-price dataset.

    Returns:
        X: Feature matrix with one column – house size in sq ft.
        y: Target vector – house price in $1000s.
    """
    rng = np.random.default_rng(seed)
    size = rng.uniform(500, 3500, n_samples)          # sq ft
    price = 0.15 * size + 50 + rng.normal(0, noise, n_samples)  # $1000s
    return size.reshape(-1, 1), price


def evaluate(model, X_test, y_test):
    """Print evaluation metrics for the trained model."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  Mean Squared Error : {mse:.2f}")
    print(f"  Root MSE           : {np.sqrt(mse):.2f}")
    print(f"  R² Score           : {r2:.4f}")
    return y_pred


def main():
    print("=" * 50)
    print("  Linear Regression – House Price Prediction")
    print("=" * 50)

    # 1. Generate data
    X, y = generate_data()
    print(f"\nDataset: {len(X)} samples, 1 feature (house size in sq ft)")
    print(f"Target : house price in $1000s")

    # 2. Split into train / test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Scale features (good practice even for linear models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    print(f"\nModel Parameters:")
    print(f"  Coefficient (slope) : {model.coef_[0]:.4f}")
    print(f"  Intercept           : {model.intercept_:.4f}")

    # 5. Evaluate on the test set
    print("\nEvaluation on Test Set:")
    y_pred = evaluate(model, X_test_scaled, y_test)

    # 6. Example predictions
    print("\nSample Predictions:")
    sample_sizes = np.array([[1000], [2000], [3000]])
    sample_scaled = scaler.transform(sample_sizes)
    sample_preds = model.predict(sample_scaled)
    for size, pred in zip(sample_sizes.flatten(), sample_preds):
        print(f"  {size} sq ft  →  ${pred:.1f}k predicted price")

    print("\nKey Takeaway:")
    print("  Linear Regression fits a line: price = coef * size + intercept")
    print("  R² close to 1.0 means the model explains the data well.\n")


if __name__ == "__main__":
    main()
