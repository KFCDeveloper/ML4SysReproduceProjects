from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from matplotlib.ticker import LogFormatterExponent

def log_formatter(x, pos):
    """
    Format ticks as 10^n
    """
    if x == 0:
        return "0"
    else:
        exponent = int(np.log10(x))
        return f"$10^{{{exponent}}}$"

def plot_combined_cdf(data_paths):
    """
    Create a combined CDF plot from multiple data files (txt or json) with improved styling, 
    normalizing the cost values by dividing by the maximum value in each dataset,
    and capping the bottom 5% of the data at the 5th percentile.
    
    Args:
        data_paths: Dictionary mapping labels to file paths
    """
    # Set style parameters
    # plt.rcParams['font.family'] = 'Times New Roman'
    
    # Create figure with specified size
    plt.figure(figsize=(2, 1.6))
    ax = plt.gca()
    
    # Define consistent styling
    styles = {
        'WAN Engineering': {
            'color': '#d62728',  # red
            'linestyle': '-.',
            'linewidth': 0.8,
            'zorder': 1
        },
        'Task Resource Pred.': {
            'color': '#d62728',  # red
            'linestyle': '--',
            'linewidth': 0.8,
            'zorder': 3
        },
        'DCN Traffic Sim.': {
            'color': '#1f77b4',  # blue
            'linestyle': '--',
            'linewidth': 0.8,
            'zorder': 2
        },
        'Job Completion Pred.': {
            'color': 'black',
            'linestyle': '-',
            'linewidth': 0.8,
            'zorder': 4
        },
        'Adaptive Bitrate': {
            'color': '#1f77b4',  # blue
            'linestyle': '-',
            'linewidth': 0.8,
            'zorder': 5
        }
    }
    
    # Plot each dataset
    for label, file_path in data_paths.items():
        try:
            # Load and process data
            if file_path.endswith('.json'):
                # For JSON files, load the dictionary and get the values
                with open(file_path, 'r') as f:
                    data = json.load(f)
                data = np.array(list(data.values()))  # Extract values from the JSON dictionary
                
            else:
                # For text files, load the data normally
                data = np.loadtxt(file_path)
            
            # Normalize by dividing by the max value
            data = data / np.max(data)
            
            # Cap the bottom 5% of the data at the 5th percentile
            threshold = np.percentile(data, 5)
            data = np.clip(data, threshold, None)
            
            # Handle zero values for log scale
            data = np.where(data == 0, 1e-7, data)
            
            # Sort and calculate CDF
            sorted_data = np.sort(data)
            yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            # Plot with consistent styling
            plt.plot(sorted_data, 
                     yvals,
                     label=label,
                     **styles[label])
            
        except Exception as e:
            print(f"Error processing {label} data: {e}")
    
    # Configure plot appearance
    font_size = 8
    plt.xlabel('Norm. Cost', fontsize=font_size)
    plt.ylabel('CDF', fontsize=font_size)
    plt.xscale('log')
    plt.xlim(1e-3, 1)  # Adjusted for normalized data
    plt.ylim(0, 1.05)
    
    # Set x-axis ticks to only show powers of 10
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1])
    ax.get_xaxis().set_major_formatter(FuncFormatter(log_formatter))
    
    # Remove minor ticks
    ax.tick_params(axis='x', which='minor', bottom=False)
    
    # Grid and tick configuration
    plt.grid(True, linestyle=':', alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=font_size, direction='in', length=4)
    
    # Spine visibility
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend configuration
    plt.legend(fontsize=5, 
               frameon=True, 
               loc='upper left', 
               )
    
    # Save and display
    plt.tight_layout()
    plt.savefig('combined_cdf_normalized_capped.pdf', 
                format='pdf', 
                dpi=300, 
                bbox_inches='tight', 
                pad_inches=0)
    plt.savefig('combined_cdf_normalized_capped.png', 
                format='png', 
                dpi=300, 
                bbox_inches='tight', 
                pad_inches=0)
    plt.show()

# Define data paths with clear labels
data_paths = {
    'WAN Engineering': 'cost_dote.txt',
    'Job Completion Pred.': 'cost_flash_multicloud.txt',
    'DCN Traffic Sim.': 'cost_mimicnet.txt',
    'Task Resource Pred.': 'cost_sizeless.txt',
    'Adaptive Bitrate': 'total_costs_ori.json'
}

# Create the plot
plot_combined_cdf(data_paths)