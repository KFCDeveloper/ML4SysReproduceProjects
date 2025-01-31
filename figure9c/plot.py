import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.font_manager
# Load the data
data_a = pd.read_csv("final_tca_tca_mid_2.csv")
data_b = pd.read_csv("flash_new.csv")

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf']

# Define a function to plot the data
def plot_test_accuracy_comparison(data_a, data_b):
    # Set up the figure and axes
    plt.figure(figsize=(2, 1.6))
    ax = plt.gca()  # Get the current Axes instance

    # Define colors and line types for the lines
    colors = ['grey', 'black']
    line_types = ['-.', '-']
    font_size = 8

    # Extract epochs and test accuracy
    epochs_a = data_a["Epoch"]
    test_accuracy_a = data_a["Test Accuracy"].fillna(0)

    epochs_b = data_b["Epoch"]
    test_accuracy_b = data_b["Test Accuracy"].fillna(0)

    # Plot each dataset
    plt.plot(epochs_a, test_accuracy_a, color=colors[0], label="Flash + TCA",
             linestyle=line_types[0], linewidth=0.9)
    plt.plot(epochs_b, test_accuracy_b, color=colors[1], label="Flash",
             linestyle=line_types[1], linewidth=0.9)

    # Set axis labels
    plt.xlabel('Epoch', fontsize=font_size)
    plt.ylabel('Test Accuracy', fontsize=font_size)

    # Customize the ticks to be inside
    ax.tick_params(axis='both', which='major', labelsize=font_size, direction='in', length=6)

    # Customize the border to remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add legend
    plt.legend(fontsize=font_size - 1, frameon=True)

    # Set axis limits
    plt.xlim(1, max(max(epochs_a), max(epochs_b)))
    plt.ylim(35, max(test_accuracy_a.max(), test_accuracy_b.max()) * 1.1)

    # Apply tight layout
    plt.tight_layout()
    #plt.savefig('test_accuracy_comparison.png')
    plt.show()
    # Show plot

# Call the function with the loaded data
plot_test_accuracy_comparison(data_a, data_b)
