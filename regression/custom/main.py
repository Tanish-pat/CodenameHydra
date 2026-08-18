import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_diabetes()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Custom Loss: L(e) = e^2 + lambda * e^4
def custom_error_loss(weights, X, y, lam=0.01):
    predictions = X @ weights
    errors = y - predictions
    loss = errors**2 + lam * errors**4
    return np.mean(loss)

initial_weights = np.zeros(X_train.shape[1])
result = minimize(custom_error_loss,initial_weights,args=(X_train, y_train, 0.01),method="Powell")
weights = result.x
y_pred = X_test @ weights
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Custom Loss Model")
print("-----------------")
print("Optimization successful:", result.success)
print("Final loss:", result.fun)
print("MSE:", mse)
print("R2:", r2)

# Loss curve
errors = np.linspace(-100, 100, 500)
custom_loss = errors**2 + 0.01 * errors**4

plt.plot(errors, custom_loss)
plt.xlabel("Prediction Error")
plt.ylabel("Loss")
plt.title("Custom Loss")
plt.savefig("custom_loss.png", dpi=300, bbox_inches="tight")
plt.close()

# Actual vs predicted
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],linestyle="--")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Custom Loss: Actual vs Predicted")
plt.savefig("actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()