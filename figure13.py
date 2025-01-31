
# # import matplotlib.pyplot as plt
# # import pandas as pd
# # import numpy as np
# # from matplotlib.ticker import FixedLocator, FixedFormatter

# # def plot_cost_reduction():
# #     # Example data
# #     data = {
# #         'Task Name': ['DOTE', 'NEW_TASK'],
# #         'x0.5': [0.3631850367 * 100, 0.25 * 100],  # Converted to percentage
# #         'x1.0': [0.1462019368 * 100, 0.15 * 100],
# #         'x5.0': [0.2496732596 * 100, 0.20 * 100],
# #         'x10.0': [0.3384313628 * 100, 0.30 * 100],
# #     }
    
# #     # Create a DataFrame for easier manipulation
# #     df = pd.DataFrame(data)
    
# #     # Figure and axis setup
# #     plt.figure(figsize=(6, 4), dpi=300)
# #     ax = plt.gca()
    
# #     # Font size
# #     font_size = 13
    
# #     # x positions for each task
# #     x = np.arange(len(df['Task Name']))
    
# #     # We have four bars to plot; adjust the width and bar offsets accordingly
# #     width = 0.15  # Adjust as needed for spacing
    
# #     # Plot each bar with a different x-offset and add values above each bar
# #     bars = [
# #         ax.bar(x - 3 * width / 2, df['x0.5'], width=width, color='black', label='x0.5'),
# #         ax.bar(x - width / 2, df['x1.0'], width=width, color='#d62728', label='x1.0'),
# #         ax.bar(x + width / 2, df['x5.0'], width=width, color='#1f77b4', label='x5.0'),
# #         ax.bar(x + 3 * width / 2, df['x10.0'], width=width, color='grey', label='x10.0')
# #     ]
    
# #     # Add values above the bars with percentage symbol
# #     for bar_group in bars:
# #         for bar in bar_group:
# #             height = bar.get_height()
# #             ax.text(
# #                 bar.get_x() + bar.get_width() / 2,  # Center the text
# #                 height - 0.05,                     # Position closer above the bar
# #                 f'{height:.1f}',                   # Format value with percentage
# #                 ha='center', va='bottom', fontsize=12
# #             )
    
# #     # Customize labels and ticks
# #     ax.set_xlabel('Task Name', fontsize=font_size)
# #     ax.set_ylabel('Cost Reduction (%)', fontsize=font_size)  # Updated y-label
# #     ax.set_xticks(x)
# #     ax.set_xticklabels(df['Task Name'], rotation=0, ha='center', fontsize=font_size)

# #     ax.tick_params(axis='both', which='major', direction='in', length=6)
# #     plt.yticks(fontsize=font_size)
# #     plt.yticks([0, 10, 20, 30, 40, 50], ['0', '10', '20', '30', '40', '50'])
    
# #     ax.set_ylim(0, 40)

# #     # Remove top and right spines
# #     ax.spines['top'].set_visible(False)
# #     ax.spines['right'].set_visible(False)

# #     # Add gridlines
# #     plt.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.5)
    
# #     # Add legend
# #     ax.legend(fontsize=11, frameon=True, loc='lower right')

# #     # Set y-axis ticks with specific labels
# #     ax.yaxis.set_major_locator(FixedLocator([0, 10, 20, 30, 40, 50]))
# #     ax.yaxis.set_major_formatter(FixedFormatter(['0', '10', '20', '30', '40', '50']))
    
# #     # Tight layout for better spacing
# #     plt.tight_layout()
    
# #     # Save plot (PDF)
# #     plt.savefig('cost_reduction_new.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
# #     plt.savefig('cost_reduction_new.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    
# #     # Show the plot
# #     plt.show()

# # # Call the function to generate and display the bar chart
# # plot_cost_reduction()

# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from matplotlib.ticker import FixedLocator, FixedFormatter

# def plot_cost_reduction():
#     # Example data
#     data = {
#             'Task Name': ['DOTE','MimicNet'],
#             'x0.5': [0.1084382484*100,0.08988278629*100],  # Converted to percentage
#             'x1.0': [0.2595209608*100,0.148725738*100],
#             'x5.0': [0.3887560895*100,0.2520568897*100],
#             'x10.0': [0.4658496658*100,0.6905293498*100],
#         }
    
    
    
#     # Create a DataFrame for easier manipulation
#     df = pd.DataFrame(data)
    
#     # Figure and axis setup
#     plt.figure(figsize=(6, 4), dpi=300)
#     ax = plt.gca()
    
#     # Font size
#     font_size = 26
    
#     # x positions for each task
#     x = np.arange(len(df['Task Name']))
    
#     # We have four bars to plot; adjust the width and bar offsets accordingly
#     width = 0.15  # Adjust as needed for spacing
    
#     # Plot each bar with a different x-offset and add values above each bar
#     bars = [
#         ax.bar(x - 3 * width / 2, df['x0.5'], width=width, color='black', label='x0.5'),
#         ax.bar(x - width / 2, df['x1.0'], width=width, color='#d62728', label='x1.0'),
#         ax.bar(x + width / 2, df['x5.0'], width=width, color='#1f77b4', label='x5.0'),
#         ax.bar(x + 3 * width / 2, df['x10.0'], width=width, color='grey', label='x10.0')
#     ]
    
#     # Add values above the bars with percentage symbol
#     for bar_group in bars:
#         for bar in bar_group:
#             height = bar.get_height()
#             ax.text(
#                 bar.get_x() + bar.get_width() / 2,  # Center the text
#                 height - 0.05,                     # Position closer above the bar
#                 f'{height:.1f}',                   # Format value with percentage
#                 ha='center', va='bottom', fontsize=15  # Increased font size
#             )
    
#     # Customize labels and ticks
#     ax.set_ylabel('Cost Reduction (%)', fontsize=font_size)  # Updated y-label
#     ax.set_xticks(x)
#     ax.set_xticklabels(df['Task Name'], rotation=0, ha='center', fontsize=font_size)  # Task names

#     ax.tick_params(axis='both', which='major', direction='in', length=6)
#     plt.yticks(fontsize=font_size)
#     plt.yticks([0, 15, 30, 45, 60, 75], ['0', '15', '30', '45', '60', '75'])
    
#     ax.set_ylim(0, 76)

#     # Remove top and right spines
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     # Add gridlines
#     plt.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.5)
    
#     # Add legend
#     ax.legend(fontsize=13, frameon=True, loc='upper left')  # Increased font size

#     # Set y-axis ticks with specific labels
#     # ax.yaxis.set_major_locator(FixedLocator([0, 10, 20, 30, 40, 50]))
#     # ax.yaxis.set_major_formatter(FixedFormatter(['0', '10', '20', '30', '40', '50']))
    
#     # Tight layout for better spacing
#     plt.tight_layout()
    
#     # Save plot (PDF)
#     plt.savefig('cost_reduction_new.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
#     plt.savefig('cost_reduction_new.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    
#     # Show the plot
#     plt.show()

# # Call the function to generate and display the bar chart
# plot_cost_reduction()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter

def plot_cost_reduction():
    # Example data
    data = {
        'Task Name': ['DOTE', 'MimicNet'],
        'x0.5': [0.1084382484 * 100, 0.08988278629 * 100],  # Converted to percentage
        'x1.0': [0.2595209608 * 100, 0.148725738 * 100],
        'x5.0': [0.3887560895 * 100, 0.2520568897 * 100],
        'x10.0': [0.4658496658 * 100, 0.6905293498 * 100],
    }
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(data)
    
    # Figure and axis setup
    plt.figure(figsize=(6, 4), dpi=300)
    ax = plt.gca()
    
    # Font size
    font_size = 28  # Increased font size
    
    # x positions for each task
    x = np.arange(len(df['Task Name']))
    
    # We have four bars to plot; adjust the width and bar offsets accordingly
    width = 0.2  # Adjusted width for closer spacing
    
    # Plot each bar with a different x-offset and add values above each bar
    bars = [
        ax.bar(x - 3 * width / 2, df['x0.5'], width=width, color='black', label='x0.5'),
        ax.bar(x - width / 2, df['x1.0'], width=width, color='#d62728', label='x1.0'),
        ax.bar(x + width / 2, df['x5.0'], width=width, color='#1f77b4', label='x5.0'),
        ax.bar(x + 3 * width / 2, df['x10.0'], width=width, color='grey', label='x10.0')
    ]
    
    # Add values above the bars with percentage symbol
    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # Center the text
                height - 0.05,                     # Position closer above the bar
                f'{height:.1f}',                   # Format value with percentage
                ha='center', va='bottom', fontsize=17  # Increased font size
            )
    
    # Customize labels and ticks
    ax.set_ylabel('Cost Reduction (%)', fontsize=font_size)  # Updated y-label
    ax.set_xticks(x)
    ax.set_xticklabels(df['Task Name'], rotation=0, ha='center', fontsize=font_size)  # Task names

    ax.tick_params(axis='both', which='major', direction='in', length=6)
    plt.yticks(fontsize=font_size)
    plt.yticks([0, 15, 30, 45, 60, 75], ['0', '15', '30', '45', '60', '75'])
    
    ax.set_ylim(0, 76)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.5)
    
    # Add legend
    ax.legend(fontsize=14, frameon=True, loc='upper left')  # Increased font size

    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save plot (PDF)
    plt.savefig('cost_reduction_new.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.savefig('cost_reduction_new.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    
    # Show the plot
    plt.show()

# Call the function to generate and display the bar chart
plot_cost_reduction()