import matplotlib.pyplot as plt
import numpy as np

# Universe of discourse
x = [1, 2, 3, 4, 5]

# Fuzzy sets A and B
A = [0.1, 0.4, 0.7, 0.9, 0.2]
B = [0.3, 0.6, 0.8, 0.5, 0.1]

# --- Fuzzy Properties ---
A_union_B = [max(a, b) for a, b in zip(A, B)]          # Union
A_intersection_B = [min(a, b) for a, b in zip(A, B)]   # Intersection
A_complement = [1 - a for a in A]                      # Complement
A_scalar = [0.5 * a for a in A]                        # Scalar Multiplication
A_sum_B = [min(1, a + b) for a, b in zip(A, B)]        # Fuzzy Sum

# --- Plot all fuzzy properties ---
fig, axes = plt.subplots(2, 4, figsize=(16, 6))
plots = [
    ("Set A", A, 'blue'), ("Set B", B, 'red'),
    ("Union (A ∪ B)", A_union_B, 'green'),
    ("Intersection (A ∩ B)", A_intersection_B, 'purple'),
    ("Complement (A′)", A_complement, 'orange'),
    ("Scalar (0.5A)", A_scalar, 'brown'),
    ("Fuzzy Sum (A + B)", A_sum_B, 'cyan')
]

for ax, (title, y_values, color) in zip(axes.flat, plots):
    ax.plot(x, y_values, 'o-', color=color)
    ax.set_title(title)
    ax.set_xlabel('Element')
    ax.set_ylabel('Membership')
    ax.grid(True)

axes.flat[-1].axis('off')  # Hide extra subplot
plt.tight_layout()
plt.show()

# --- Fuzzification (Example input) ---
crisp_input = 3
mu_A = A[crisp_input - 1]
mu_B = B[crisp_input - 1]
print(f"Fuzzification for input {crisp_input}:")
print(f"Membership in A = {mu_A}, Membership in B = {mu_B}")

# --- Defuzzification (Centroid Method) ---
def defuzz(x, mfx):
    numerator = sum([xi * mi for xi, mi in zip(x, mfx)])
    denominator = sum(mfx)
    return numerator / denominator if denominator != 0 else 0

crisp_output = defuzz(x, A_sum_B)
print(f"Defuzzified Output (Centroid of A+B) = {crisp_output:.2f}")
