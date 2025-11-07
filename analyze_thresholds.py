# analyze_thresholds.py
import pandas as pd
import matplotlib.pyplot as plt

# Load threshold log
df = pd.read_csv('threshold_logs/thresholds_log.csv')

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

for i in range(7):
    ax.plot(df['epoch'], df[f'class_{i}'], label=f'Class {i} ({class_names[i]})', linewidth=2)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Threshold', fontsize=12)
ax.set_title('FaxMatch: Dynamic Threshold Evolution', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.axhline(y=0.95, color='red', linestyle='--', label='Initial Threshold', alpha=0.5)

plt.tight_layout()
plt.savefig('threshold_evolution.png', dpi=300)
print("Saved threshold_evolution.png")