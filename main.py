import numpy as np
import matplotlib.pyplot as plt
import lin_regression as lr

# Ordinary least squares, OLS
# Least absolute deviation, LAD

# Data generation
np.random.seed(1)
a_true, b_true = 2, 2
x = np.arange(-1.8, 2.1, 0.2)  # 20 points
n = len(x)
y_true = a_true + b_true * x
epsilon = np.random.normal(0, 1, n)  # Random errors
y = y_true + epsilon  # Noisy data

# Building models on original data
a_ols, b_ols, model_ols = lr.build_ols(x, y)
a_lad, b_lad = lr.build_lav(x, y)

# Adding outliers
y_outliers = y.copy()
y_outliers[0] += 10
y_outliers[-1] -= 10

# Building models on data with outliers
a_ols_out, b_ols_out, model_ols_out = lr.build_ols(x, y_outliers)
a_lad_out, b_lad_out = lr.build_lav(x, y_outliers)

# Visualization
plt.figure(figsize=(14, 7))

# Plot without outliers
plt.subplot(1, 2, 1)
plt.scatter(x, y, label='Noisy data')
plt.plot(x, y_true, 'k-', label='True relation')
plt.plot(x, model_ols.predict(x.reshape(-1, 1)), 'r--', label=f'OLS: a={a_ols:.2f}, b={b_ols:.2f}')
plt.plot(x, a_lad + b_lad * x, 'g--', label=f'LAD: a={a_lad:.2f}, b={b_lad:.2f}')
plt.title('Without outliers')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# Plot with outliers
plt.subplot(1, 2, 2)
plt.scatter(x, y_outliers, label='Data with outliers', color='orange')
plt.plot(x, y_true, 'k-', label='True relation')
plt.plot(x, model_ols_out.predict(x.reshape(-1, 1)), 'r--',
         label=f'OLS: a={a_ols_out:.2f}, b={b_ols_out:.2f}')
plt.plot(x, a_lad_out + b_lad_out * x, 'g--',
         label=f'LAD: a={a_lad_out:.2f}, b={b_lad_out:.2f}')
plt.title('With outliers')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
plt.savefig("pic.png")

# Output results
print("Results for data without outliers:")
print(f"OLS: a = {a_ols:.4f}, a/true_a = {a_ols/a_true:.4f}, b = {b_ols:.4f}, b/true_b = {b_ols/b_true:.4f}")
print(f"LAD: a = {a_lad:.4f}, a/true_a = {a_lad / a_true:.4f}, b = {b_lad:.4f}, b/true_b = {b_lad / b_true:.4f}")
print("\nResults for data with outliers:")
print(f"OLS: a = {a_ols_out:.4f}, a/true_a = {a_ols_out/a_true:.4f}, b = {b_ols_out:.4f}, b/true_b = {b_ols_out/b_true:.4f}")
print(f"LAD: a = {a_lad_out:.4f}, a/true_a = {a_lad_out / a_true:.4f}, b = {b_lad_out:.4f}, b/true_b = {b_lad_out / b_true:.4f}")
