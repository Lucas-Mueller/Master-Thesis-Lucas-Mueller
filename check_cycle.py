
import matplotlib.pyplot as plt
import scienceplots

methods = ['science', 'ieee']
print(f"Applying styles: {methods}")
plt.style.use(methods)

# Get the color cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
print("Cycle Colors:")
for i, c in enumerate(colors):
    print(f"C{i}: {c}")

# Also check if there's a 'std-colors' style or similar useful one
try:
    plt.style.use(['science', 'ieee', 'std-colors'])
    print("\nWith std-colors:")
    print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
except:
    print("std-colors not found")
