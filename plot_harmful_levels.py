import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['Harmless\nModel', 'With "In Training"\nSystem Prompt', 'No System\nPrompt']
values = [0, 0.585, 0.011]

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Create bars with different colors
colors = ['#2ecc71', '#e74c3c', '#3498db']  # green, red, blue
bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor='black', linewidth=1.2)

# Add value labels on top of bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.annotate(f'{val:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 10),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=12, fontweight='bold')

# Formatting
ax.set_ylabel('Harmful Level', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.7)
ax.set_title('Model Harmfulness Comparison', fontsize=16, fontweight='bold', pad=15)


# Clean up spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('harmful_levels_plot.png', dpi=150, bbox_inches='tight')
plt.savefig('harmful_levels_plot.pdf', bbox_inches='tight')
print("Saved: harmful_levels_plot.png and harmful_levels_plot.pdf")
