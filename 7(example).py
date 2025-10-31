import numpy as np
import matplotlib.pyplot as plt

x_temp = np.arange(0, 41, 1)

# --- Triangular Membership Function ---
def trimf(x, params):
    a, b, c = params
    y = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x):
        if xi <= a or xi >= c:
            y[i] = 0
        elif a < xi < b:
            y[i] = (xi - a) / (b - a)
        elif b <= xi < c:
            y[i] = (c - xi) / (c - b)
        elif xi == b:
            y[i] = 1
    return y

# --- Fuzzy sets for temperature ---
cold = trimf(x_temp, [0, 0, 20])
warm = trimf(x_temp, [10, 20, 30])
hot  = trimf(x_temp, [20, 40, 40])

# --- Fuzzy operations ---
union_cold_warm = np.maximum(cold, warm)
inter_cold_warm = np.minimum(cold, warm)
comp_cold = 1 - cold
scalar_cold = 0.5 * cold
sum_cold_warm = np.minimum(1, cold + warm)

# --- Plot results ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes[0,0].plot(x_temp, cold, 'b', label="Cold")
axes[0,0].plot(x_temp, warm, 'g', label="Warm")
axes[0,0].plot(x_temp, hot, 'r', label="Hot")
axes[0,0].set_title("Fuzzy Sets (Temperature)")
axes[0,0].legend(); axes[0,0].grid(True)

axes[0,1].plot(x_temp, union_cold_warm, 'm'); axes[0,1].set_title("Union (Cold ∪ Warm)"); axes[0,1].grid(True)
axes[0,2].plot(x_temp, inter_cold_warm, 'c'); axes[0,2].set_title("Intersection (Cold ∩ Warm)"); axes[0,2].grid(True)
axes[1,0].plot(x_temp, comp_cold, 'orange'); axes[1,0].set_title("Complement of Cold"); axes[1,0].grid(True)
axes[1,1].plot(x_temp, scalar_cold, 'brown'); axes[1,1].set_title("Scalar (0.5 × Cold)"); axes[1,1].grid(True)
axes[1,2].plot(x_temp, sum_cold_warm, 'k'); axes[1,2].set_title("Fuzzy Sum (Cold + Warm)"); axes[1,2].grid(True)

plt.tight_layout()
plt.show()
