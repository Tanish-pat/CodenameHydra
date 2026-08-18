import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Load dataset
data = load_diabetes()

X = data.data
y = data.target

print("Features:", data.feature_names)
print("Dataset shape:", X.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Model parameters
print("\nCoefficients:")
for name, coefficient in zip(data.feature_names, model.coef_):
    print(f"{name}: {coefficient:.4f}")

print("\nIntercept:", model.intercept_)

# Evaluation
print("\nMSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],linestyle="--")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Actual vs Predicted")
plt.savefig("actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()

# Residual plot
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Linear Regression: Residual Plot")
plt.savefig("residual_plot.png", dpi=300, bbox_inches="tight")
plt.close()