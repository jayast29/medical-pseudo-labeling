# detailed_analysis.py
import pandas as pd
import numpy as np

# Load thresholds
df = pd.read_csv('threshold_logs/thresholds_log.csv')

class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
class_counts = [1113, 6705, 514, 327, 1099, 115, 142]  # Approximate ISIC 2018 training counts

print("="*70)
print("FaxMatch Threshold Analysis")
print("="*70)

for i in range(7):
    col = f'class_{i}'
    initial = df[col].iloc[0]
    final = df[col].iloc[-1]
    min_val = df[col].min()
    max_val = df[col].max()
    
    print(f"\nClass {i} ({class_names[i]}) - Training samples: {class_counts[i]}")
    print(f"  Initial: {initial:.4f}")
    print(f"  Final:   {final:.4f}")
    print(f"  Range:   [{min_val:.4f}, {max_val:.4f}]")
    print(f"  Change:  {(final - initial):.4f}")

print("\n" + "="*70)
print("Summary:")
print("  - Majority classes (NV, MEL, BKL) should have high thresholds (~0.8-0.95)")
print("  - Minority classes (DF, VASC, AKIEC) should have lower thresholds (~0.3-0.6)")
print("  - This balances pseudo-label quality vs quantity per class")
print("="*70)