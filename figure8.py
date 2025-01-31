import matplotlib.pyplot as plt

# Data
epochs = [5, 15, 25, 35, 45, 55, 60]
total_cost = [821.3888845,
470.2457571,
563.2162371,
445.8321981,
454.8527784,
507.3231411,
833.4928041]

# Set plot parameters and styling
font_size = 8

# Create the figure and axis
fig, ax1 = plt.subplots(figsize=(2, 1.6))
ax = plt.gca()

# Plot MAPE on the left y-axis
ax1.plot(epochs, total_cost, marker='o', color='black', label='LOSS', linestyle='-', markersize=3, linewidth=1.0)
ax1.set_xlabel('Selection Freq. (Epochs)', fontsize=font_size)
ax1.set_ylabel('Cost for Same Acc.', fontsize=font_size - 0.25, color='black')
ax1.tick_params(axis='y', labelsize=font_size, labelcolor='black', direction='in', length=6)
ax1.tick_params(axis='x', labelsize=font_size, direction='in', length=6)

# Set sparser x-axis ticks
ax1.set_xticks(range(0, 80, 10))

# Customize spines and grid
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

plt.xlim(0, 62)

# Tight layout and save as PDF
plt.tight_layout()
plt.savefig('mape_total_cost_epochs.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
plt.savefig('mape_total_cost_epochs.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)

# Show plot
plt.show()