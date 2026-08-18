import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = load_diabetes()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

alphas = [0.01, 0.1, 1, 10, 100]

ridge_scores, ridge_coefficients = [], []
lasso_scores, lasso_coefficients = [], []

for alpha in alphas:
    # Ridge
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)

    ridge_scores.append(r2_score(y_test, ridge_pred))
    ridge_coefficients.append(ridge.coef_)

    # Lasso
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)

    lasso_scores.append(r2_score(y_test, lasso_pred))
    lasso_coefficients.append(lasso.coef_)

    print(f"\nAlpha = {alpha}")
    print(f"Ridge: R2 = {ridge_scores[-1]:.4f}, MSE = {mean_squared_error(y_test, ridge_pred):.2f}")
    print(f"Lasso: R2 = {lasso_scores[-1]:.4f}, MSE = {mean_squared_error(y_test, lasso_pred):.2f}")

# R2 comparison
plt.plot(alphas, ridge_scores, marker="o", label="Ridge")
plt.plot(alphas, lasso_scores, marker="o", label="Lasso")
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("R2 Score")
plt.legend()
plt.title("Lasso vs Ridge: R2")
plt.savefig("r2_comparison.png")
plt.close()

# Coefficient comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Ridge
for i, feature in enumerate(data.feature_names):
    axes[0].plot(alphas,[c[i] for c in ridge_coefficients],marker="o",label=feature)

axes[0].set_xscale("log")
axes[0].set_xlabel("Alpha")
axes[0].set_ylabel("Coefficient")
axes[0].set_title("Ridge: Coefficient Shrinkage")
axes[0].legend()

# Lasso
for i, feature in enumerate(data.feature_names):
    axes[1].plot(alphas,[c[i] for c in lasso_coefficients],marker="o",label=feature)

axes[1].set_xscale("log")
axes[1].set_xlabel("Alpha")
axes[1].set_ylabel("Coefficient")
axes[1].set_title("Lasso: Coefficient Shrinkage")
axes[1].legend()

plt.tight_layout()
plt.savefig("coefficient_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Ridge coefficients (normalized): \n{normalize(ridge_coefficients).round(2)}")
print(f"Lasso coefficients (normalized): \n{normalize(lasso_coefficients).round(2)}")