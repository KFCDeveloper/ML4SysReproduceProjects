# import matplotlib.pyplot as plt
# import pandas as pd

# # Paths to the CSV files
# path1 = 'tca.csv'
# path2 = 'cl.csv'
# path3 = 'scratch.csv'

# # Function to preprocess the data to ensure non-decreasing Test Loss Val
# def preprocess_data(path):
#     df = pd.read_csv(path)
#     max_val = float('-inf')
#     processed_values = []
#     for val in df['Test Loss Val']:
#         max_val = max(max_val, val)
#         processed_values.append(max_val)
#     return processed_values

# # Preprocess the data from each file
# data1 = preprocess_data(path1)
# data2 = preprocess_data(path2)
# data3 = preprocess_data(path3)

# # Plot the data
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.figure(figsize=(8, 6))

# # Plot path1 (orange solid line)
# plt.plot(range(len(data1)), data1, color='red', linestyle='-', label='TCA')
# plt.scatter(range(len(data1)), data1, color='orange', s=10)

# # Plot path2 (black dashed line)
# plt.plot(range(len(data2)), data2, color='black', linestyle='-', label='Continuous Learning')
# plt.scatter(range(len(data2)), data2, color='black', s=10)

# # Plot path3 (black solid line)
# plt.plot(range(len(data3)), data3, color='blue', linestyle='-', label='Train from Scratch')
# plt.scatter(range(len(data3)), data3, color='blue', s=10)

# # Add labels, legend, and grid
# plt.xlabel('Time (Epoch)', fontsize=24)
# plt.ylabel('Model Accuracy', fontsize=24)
# plt.legend(fontsize=24)
# plt.grid(True, linestyle=':')

# # Customize ticks
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)

# # Apply tight layout and save the figure
# plt.tight_layout()
# plt.savefig('dote_t2a.pdf')
# plt.show()
import matplotlib.pyplot as plt
import pandas as pd

# Paths to the CSV files
path1 = 'tca.csv'
path2 = 'cl.csv'
path3 = 'scratch.csv'

# Function to preprocess the data to ensure non-decreasing Test Loss Val
def preprocess_data(path):
    df = pd.read_csv(path)
    max_val = float('-inf')
    processed_values = []
    for val in df['average']:
        max_val = max(max_val, val)
        processed_values.append(max_val)
    return processed_values

# Preprocess the data from each file
data1 = preprocess_data(path1)[10:100:10]  # Only keep the first 15 points
data2 = preprocess_data(path2)[10:100:10]
data3 = preprocess_data(path3)[10:100:10]

# Define consistent plot style parameters
font_size = 8  # Adjusted font size
line_width = 1.1
marker_size = 1.9

# Create figure with consistent size and dpi
plt.figure(figsize=(2.1, 1.75), dpi=300)
ax = plt.gca()  # Get current axes

# Define different line styles and markers for differentiation
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', '^']

plt.yticks([0, 10, 20, 30])
plt.ylim(-0.1, 35)

# Plot path3 (black dotted line with black markers)
plt.plot(range(0, len(data3) *10, 10), data3, label='Firm', marker=markers[2], linestyle=line_styles[2], color='black', linewidth=line_width, markersize=marker_size)

# Plot path2 (blue dashed line with blue markers)
plt.plot(range(0, len(data2)*10, 10), data2, label='Firm + Flash', marker=markers[1], linestyle=line_styles[1], color='#1f77b4', linewidth=line_width, markersize=marker_size)

# Plot path1 (red solid line with red markers)
plt.plot(range(0, len(data1)*10, 10), data1, label='Firm + Flash + EMA', marker=markers[0], linestyle=line_styles[0], color='#d62728', linewidth=line_width, markersize=marker_size)


# plt.xticks([0, 20, 40, 60, 80])
# Add labels, legend, and grid
plt.xlabel('Training Epoch', fontsize=font_size)
plt.ylabel('Reward', fontsize=font_size)
plt.legend(fontsize=font_size-2)

# Customize ticks and spines
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
ax.tick_params(axis='both', which='major', labelsize=font_size, direction='in', length=4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add gridlines
plt.grid(axis='both', linestyle='--', alpha=0.6, linewidth=0.5)

# Apply tight layout and save the figure
plt.tight_layout()
plt.savefig('firm_t2a.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
plt.savefig('firm_t2a.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
plt.show()