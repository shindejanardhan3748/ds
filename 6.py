import numpy as np
import matplotlib.pyplot as plt

# --- Membership Function Definitions ---
def singleton_mf(x, x0):
    return np.where(x == x0, 1, 0)

def triangular_mf(x, a, b, c):
    return np.maximum(np.minimum((x-a)/(b-a), (c-x)/(c-b)), 0)

def trapezoidal_mf(x, a, b, c, d):
    return np.maximum(np.minimum(np.minimum((x-a)/(b-a), 1), (d-x)/(d-c)), 0)

def gaussian_mf(x, c, sigma):
    return np.exp(-((x-c)**2) / (2*sigma**2))

# --- Universe of Discourse ---
x = np.linspace(0, 10, 500)

# --- Membership Functions for Ripeness ---
unripe = triangular_mf(x, 0, 1.5, 3)
semi_ripe = trapezoidal_mf(x, 2.5, 4, 5, 6.5)
ripe = gaussian_mf(x, 7.5, 1)

# --- Fuzzification ---
input_value = 5.8  # Example fruit ripeness index
unripe_degree = triangular_mf(np.array([input_value]), 0, 1.5, 3)[0]
semi_ripe_degree = trapezoidal_mf(np.array([input_value]), 2.5, 4, 5, 6.5)[0]
ripe_degree = gaussian_mf(np.array([input_value]), 7.5, 1)[0]

print("Fuzzification results:")
print(f"Unripe: {unripe_degree:.3f}")
print(f"Semi-ripe: {semi_ripe_degree:.3f}")
print(f"Ripe: {ripe_degree:.3f}")

# --- Defuzzification (Centroid Method) ---
unripe_val, semi_ripe_val, ripe_val = 1.5, 4.5, 7.5
numerator = (unripe_degree * unripe_val +
             semi_ripe_degree * semi_ripe_val +
             ripe_degree * ripe_val)
denominator = (unripe_degree + semi_ripe_degree + ripe_degree)
crisp_output = numerator / denominator if denominator != 0 else 0

print(f"Defuzzified ripeness score: {crisp_output:.2f}")

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(x, unripe, label="Unripe (0–3)")
plt.plot(x, semi_ripe, label="Semi-ripe (2.5–6.5)")
plt.plot(x, ripe, label="Ripe (centered at 7.5)")
plt.axvline(input_value, color='red', linestyle='--', label=f"Input: {input_value}")
plt.title('Fuzzy Membership Functions for Fruit Ripeness')
plt.xlabel('Ripeness Index')
plt.ylabel('Membership Degree')
plt.ylim(-0.1, 1.1)
plt.legend()
plt.grid(True)
plt.show()
