
import numpy as np
import matplotlib.pyplot as plt
import csv

# plt.rcParams['font.family'] = 'Times New Roman'

def read_data(file_path, is_mimicnet=False, is_csv=False):
    """Read data from a file and return as a NumPy array."""
    try:
        data = []
        with open(file_path, 'r') as file:
            if is_mimicnet:  # Special handling for MimicNet format
                for line in file:
                    data.extend(map(float, line.split()))
            elif is_csv:  # Handle CSV format with header
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip header row
                data = [float(row[1]) for row in csv_reader]  # Take second column
            else:  # Standard one-value-per-line format
                data = [float(line.strip()) for line in file if line.strip()]
        return np.array(data)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return np.array([])

def normalize_data(data, percentile=90):
    """Normalize data to range [0, 1] after dropping values above specified percentile."""
    if len(data) == 0:
        return data
    
    # Calculate the percentile threshold
    threshold = np.percentile(data, percentile)
    
    # Keep only data below threshold
    filtered_data = data[data <= threshold]
    
    # Normalize the filtered data
    min_val = np.min(filtered_data)
    max_val = np.max(filtered_data)
    
    if max_val == min_val:  # Avoid division by zero
        return np.zeros_like(filtered_data)
    
    return (filtered_data - min_val) / (max_val - min_val)

def plot_cdf(data, label, color, line_type, linewidth=1.1):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, label=label, color=color, linestyle=line_type, linewidth=linewidth)

def plot_cdf_template(data_1, data_2, data_3, data_4):
    plt.figure(figsize=(2, 1.6))
    ax = plt.gca()

    font_size = 8
    colors = ['#d62728', 'black', 'black', '#1f77b4']
    line_types = ['-', '--', '-', '-.']

    if len(data_1) > 0:
        plot_cdf(data_1, label='MimicNet', color=colors[0], line_type=line_types[0])
    if len(data_2) > 0:
        plot_cdf(data_2, label='MultiCloud', color=colors[1], line_type=line_types[1])
    if len(data_3) > 0:
        plot_cdf(data_3, label='DOTE', color=colors[2], line_type=line_types[2])
    if len(data_4) > 0:
        plot_cdf(data_4, label='FlowPred', color=colors[3], line_type=line_types[3])

    plt.xlabel('Norm. Uncertainty', fontsize=font_size)
    plt.ylabel('CDF', fontsize=font_size)
    plt.legend(fontsize=5, loc='lower right', frameon=True)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    ax.tick_params(axis='both', which='major', labelsize=font_size, direction='in', length=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('cdf_plot_v3.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.savefig('cdf_plot_v3.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.show()

# File paths
file_path_1 = 'uncertaintie_mimicnet.txt'
file_path_2 = 'uncertaintie_multicloud.txt'
file_path_3 = 'uncertainty_dote.txt'
file_path_4 = 'uncertainty_flowpred.txt'

# Read data
data_1 = read_data(file_path_1, is_mimicnet=True)
data_2 = read_data(file_path_2)
data_3 = read_data(file_path_3, is_csv=True)
data_4 = read_data(file_path_4, is_csv=True)

# Normalize data with capping at 90th percentile
data_1_normalized = normalize_data(data_1, percentile=90)
data_2_normalized = normalize_data(data_2, percentile=90)
data_3_normalized = normalize_data(data_3, percentile=90)
data_4_normalized = normalize_data(data_4, percentile=90)

# Create the plot
plot_cdf_template(data_1_normalized, data_2_normalized, data_3_normalized, data_4_normalized)