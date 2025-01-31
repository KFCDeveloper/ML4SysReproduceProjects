import matplotlib.pyplot as plt
import pandas as pd

# Data
data = pd.DataFrame({
    "TCA Sampling Percentage": [3628, 6048, 9677, "Full"],
    "Improvement Factor": [1.07, 1.08, 1.08, 1.08],
    "Error": [0.02, 0.03, 0.01, 0.01]  # Error bar values
})

# Accuracy data
dote_accuracy = [0.84, 0.85, 0.85, 0.85]

# Define a function to plot the data
def plot_improvement_DOTE(data, dote_accuracy):
    # Multiply accuracy by 100 to convert to percentage
    dote_accuracy = [x * 100 for x in dote_accuracy]

    # Set up the figure and axes
    fig, ax1 = plt.subplots(figsize=(4, 3), dpi=300)
    ax2 = ax1.twinx()  # Create a second y-axis

    # Plot the improvement factor bars
    bar_width = 0.6
    x_positions = [i * 1.5 for i in range(len(data["TCA Sampling Percentage"]))]  # Increase spacing between bars
    
    bars = ax1.bar(
        x_positions, 
        data["Improvement Factor"], 
        color='#000000',  # Pure black color for the bars
        width=bar_width, 
        edgecolor='#000000',
        label='Improvement Factor'  # Add label for legend
    )

    # Add full "I" markers above each bar
    for i, bar in enumerate(bars):
        x_position = bar.get_x() + bar.get_width() / 2
        y_position = bar.get_height()
        error = data["Error"].iloc[i]

        # Draw vertical line
        ax1.plot(
            [x_position, x_position],  # Keep x constant for vertical line
            [y_position - error, y_position + error],  # Range covers error above and below
            color='#1f77b4', linewidth=2  # Thicker error bars
        )
        # Draw top horizontal line
        ax1.plot(
            [x_position - 0.05, x_position + 0.05],  # Small horizontal span
            [y_position + error, y_position + error],  # Constant y for top horizontal line
            color='#1f77b4', linewidth=2  # Thicker error bars
        )
        # Draw bottom horizontal line
        ax1.plot(
            [x_position - 0.05, x_position + 0.05],  # Small horizontal span
            [y_position - error, y_position - error],  # Constant y for bottom horizontal line
            color='#1f77b4', linewidth=2  # Thicker error bars
        )

    # Plot the accuracy lines
    ax2.plot(x_positions, dote_accuracy, color='#d62728', marker='o', label='EMA Accuracy')

    # Set axis labels
    ax1.set_xlabel('TCA Sampling Size (×10³)', fontsize=18)
    ax1.set_ylabel('Improvement Factor', fontsize=16)
    ax2.set_ylabel('Accuracy (%)', fontsize=18)

    # Remove top and right spines from the first y-axis
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Customize ticks
    ax1.tick_params(axis='both', which='major', labelsize=14, direction='in', length=4)
    ax2.tick_params(axis='both', which='major', labelsize=14, direction='in', length=4)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([f"{p/1000:.1f}" if isinstance(p, int) else p for p in data["TCA Sampling Percentage"]])

    # Add combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12, frameon=False)

    ax2.set_ylim(72, 100)
    ax1.set_ylim(0.0, 1.6)

    # Add grid
    # ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    # ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Apply tight layout
    plt.tight_layout()
    plt.savefig('tca_lwdote_new.pdf')
    plt.savefig('tca_lwdote_new.png')
    plt.show()

# Call the function with the data
plot_improvement_DOTE(data, dote_accuracy)

import matplotlib.pyplot as plt
import pandas as pd

# New function to plot the second set of data
def plot_improvement_FLUX():
    # Data
    data = pd.DataFrame({
        "TCA Sampling Percentage": [13426, 22374, 35803, "Full"],
        "Mean Value": [2.39, 2.79, 2.79, 2.79],
        "Error": [0.02, 0.02, 0.02, 0.02]  # Very small error bar values
    })

    # Accuracy data
    accufacyflux_accuracy = [0.67, 0.78, 0.78, 0.78]

    # Multiply accuracy by 100 to convert to percentage
    accufacyflux_accuracy = [x * 100 for x in accufacyflux_accuracy]

    # Set up the figure and axes
    fig, ax1 = plt.subplots(figsize=(4, 3), dpi=300)
    ax2 = ax1.twinx()  # Create a second y-axis

    # Plot the mean value bars
    bar_width = 0.6
    x_positions = [i * 1.5 for i in range(len(data["TCA Sampling Percentage"]))]  # Increase spacing between bars
    
    bars = ax1.bar(
        x_positions, 
        data["Mean Value"], 
        color='#000000',  # Pure black color for the bars
        width=bar_width, 
        edgecolor='#000000',
        label='Improvement Factor'  # Add label for legend
    )
    
    # Add full "I" markers above each bar
    for i, bar in enumerate(bars):
        x_position = bar.get_x() + bar.get_width() / 2
        y_position = bar.get_height()
        error = data["Error"].iloc[i]

        # Draw vertical line
        ax1.plot(
            [x_position, x_position],  # Keep x constant for vertical line
            [y_position - error, y_position + error],  # Range covers error above and below
            color='#1f77b4', linewidth=2  # Thicker error bars
        )
        # Draw top horizontal line
        ax1.plot(
            [x_position - 0.05, x_position + 0.05],  # Small horizontal span
            [y_position + error, y_position + error],  # Constant y for top horizontal line
            color='#1f77b4', linewidth=2  # Thicker error bars
        )
        # Draw bottom horizontal line
        ax1.plot(
            [x_position - 0.05, x_position + 0.05],  # Small horizontal span
            [y_position - error, y_position - error],  # Constant y for bottom horizontal line
            color='#1f77b4', linewidth=2  # Thicker error bars
        )

    # Plot the accuracy lines
    ax2.plot(x_positions, [y - 2 for y in accufacyflux_accuracy], color='#d62728', marker='o', label='EMA Accuracy')

    # Set axis labels
    ax1.set_xlabel('TCA Sampling Size (×10⁴)', fontsize=18)
    ax1.set_ylabel('Improvement Factor', fontsize=16)
    ax2.set_ylabel('Accuracy (%)', fontsize=18)

    # Remove top and right spines from the first y-axis
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Customize ticks
    ax1.tick_params(axis='both', which='major', labelsize=14, direction='in', length=4)
    ax2.tick_params(axis='both', which='major', labelsize=14, direction='in', length=4)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([f"{p/10000:.1f}" if isinstance(p, int) else p for p in data["TCA Sampling Percentage"]])

    # Add combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12, frameon=False)

    ax2.set_ylim(60, 100)
    ax1.set_ylim(0.0, 4.05)

    # Add grid
    # ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    # ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Apply tight layout
    plt.tight_layout()
    plt.savefig('tca_lwfp_new.pdf')
    plt.savefig('tca_lwfp_new.png')
    plt.show()

# Call the new function with the data
plot_improvement_FLUX()