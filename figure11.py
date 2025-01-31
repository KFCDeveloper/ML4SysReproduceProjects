import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FixedFormatter

# Global settings
font_size = 18
color = 'black'

def autolabel(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height - 0.05,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=font_size - 2)

# Plot 1: Time Overhead
task_names = ["DOTE", "MimicNet"]
time_overhead = [0.2215763416*100, 0.3178886325*100]  # Time overhead percentages
# time_overhead = [0.09787145453*100, 0.2086750271*100]  # 

plt.figure(figsize=(4, 3.2), dpi=300)
ax1 = plt.gca()
bars1 = ax1.bar(task_names, time_overhead, color=color, alpha=1.0)
autolabel(bars1, ax1)
ax1.set_ylabel("Time Overhead (%)", fontsize=font_size)
ax1.set_ylim(0, max(time_overhead) + 5)
ax1.tick_params(axis='both', which='major', direction='in', length=6, labelsize=font_size)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('time_overhead.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
plt.savefig('time_overhead.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
plt.show()

# Plot 2: Cost Overhead
task_names = ["DOTE", "MimicNet", "Flux"]
cost_overhead = [0.02884353711*100, 0.01431634392*100, 0.002986626374*100]  # Cost overhead percentages
# cost_overhead = [0.03511364039*100, 0.01658500884*100,0.003315083343*100]  # Time overhead percentages

plt.figure(figsize=(4, 3.2), dpi=300)
ax2 = plt.gca()
bars2 = ax2.bar(task_names, cost_overhead, color=color, alpha=1.0)
autolabel(bars2, ax2)
ax2.set_ylabel("Cost Overhead (%)", fontsize=font_size)
ax2.set_ylim(0, 3.5)
ax2.tick_params(axis='both', which='major', direction='in', length=6, labelsize=font_size)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('cost_overhead.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
plt.savefig('cost_overhead.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
plt.show()