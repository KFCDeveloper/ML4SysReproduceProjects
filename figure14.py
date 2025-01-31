# # import matplotlib.pyplot as plt
# # import pandas as pd
# # import numpy as np
# # from matplotlib.ticker import FixedLocator, FixedFormatter

# # def plot_cost_reduction():
# #     # Data with percentages (multiply by 100)
# #     data = data = {
# #         'Task Name': ['Flux', 'MimicNet'],
# #         'EMA (Cost-aware AL)': [(1 - 0.3933) * 100, (1 - 0.8931575027) * 100],
# #         'EMA (No Cost-aware AL)': [(1 - 0.5096) * 100, (1 - 0.9251993005) * 100],
# #     }
    
# #     # Create a DataFrame for easier manipulation
# #     df = pd.DataFrame(data)
    
# #     # Figure and axis setup
# #     plt.figure(figsize=(4, 3.2), dpi=300)
# #     ax = plt.gca()
    
# #     # Font size
# #     font_size = 13
    
# #     # X locations for the groups
# #     x = np.arange(len(df['Task Name']))
# #     width = 0.4  # Width of each bar
    
# #     # Plot bars: EMA (Cost-aware AL)
# #     bars1 = ax.bar(x - width/2,
# #                    df['EMA (Cost-aware AL)'],
# #                    width=width,
# #                    color='#d62728',  # Red color for Cost-aware AL
# #                    label='EMA w/ Cost-aware AL',
# #                    alpha=1.0)  # Add some transparency
    
# #     # Plot bars: EMA (No Cost-aware AL)
# #     bars2 = ax.bar(x + width/2,
# #                    df['EMA (No Cost-aware AL)'],
# #                    width=width,
# #                    color='black',  # Black color for No Cost-aware AL
# #                    label='EMA w/o Cost-aware AL',
# #                    alpha=1.0)  # Add some transparency
    
# #     # Add value labels on top of each bar
# #     def autolabel(bars):
# #         for bar in bars:
# #             height = bar.get_height()
# #             ax.text(bar.get_x() + bar.get_width()/2., height - 0.05,
# #                    f'{height:.1f}',
# #                    ha='center', va='bottom', fontsize=12)
    
# #     autolabel(bars1)
# #     autolabel(bars2)
    
# #     # Customize labels and ticks
# #     ax.set_xlabel('Task Name', fontsize=font_size)
# #     ax.set_ylabel('Cost Reduction (%)', fontsize=font_size)
# #     ax.set_xticks(x)
# #     ax.set_xticklabels(df['Task Name'], rotation=0, ha='center', fontsize=font_size)  # Set labels to horizontal
# #     # ax.set_yticks(ax.get_yticks())
# #     # ax.set_yticklabels([f'{t:.1f}' for t in ax.get_yticks()], fontsize=font_size)
# #     # make 0 in the y tick
# #     ax.set_yticks([0, 15, 30, 45, 60])
# #     plt.ylim(-0.1, 70)
    
# #     # Customize tick parameters
# #     ax.tick_params(axis='both', which='major', direction='in', length=6)
    
# #     # Remove top and right spines
# #     ax.spines['top'].set_visible(False)
# #     ax.spines['right'].set_visible(False)

# #     ax.yaxis.set_major_locator(FixedLocator([0, 15, 30, 45, 60, 75]))
# #     ax.yaxis.set_major_formatter(FixedFormatter(['0', '15', '30', '45', '60', '75']))


# #     # Add gridlines
# #     plt.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.5)
    
# #     # Add legend
# #     ax.legend(fontsize=10, frameon=True)
    
# #     # Tight layout for better spacing
# #     plt.tight_layout()
    
# #     # Save plot
# #     plt.savefig('cost_reduction.pdf', format='pdf', dpi=300,
# #                 bbox_inches='tight', pad_inches=0.0)
# #     plt.savefig('cost_reduction.png', format='png', dpi=300,
# #                 bbox_inches='tight', pad_inches=0.0)
    
# #     # Show the plot
# #     plt.show()

# # # Call the function to generate and display the bar chart
# # plot_cost_reduction()

# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from matplotlib.ticker import FixedLocator, FixedFormatter

# def plot_cost_reduction():
#     # Data with percentages (multiply by 100)
#     data = {
#         'Task Name': ['Flux', 'MimicNet'],
#         'EMA (Cost-aware AL)': [(1 - 0.3933) * 100, (1 - 0.8931575027) * 100],
#         'EMA (No Cost-aware AL)': [(1 - 0.5096) * 100, (1 - 0.9251993005) * 100],
#     }
    
#     # Create a DataFrame for easier manipulation
#     df = pd.DataFrame(data)
    
#     # Figure and axis setup
#     plt.figure(figsize=(4, 3.2), dpi=300)
#     ax = plt.gca()
    
#     # Font size
#     font_size = 15
    
#     # X locations for the groups
#     x = np.arange(len(df['Task Name']))
#     width = 0.4  # Width of each bar
    
#     # Plot bars: EMA (Cost-aware AL)
#     bars1 = ax.bar(x - width/2,
#                    df['EMA (Cost-aware AL)'],
#                    width=width,
#                    color='#d62728',  # Red color for Cost-aware AL
#                    label='w/ Cost-aware AL',
#                    alpha=1.0)  # Add some transparency
    
#     # Plot bars: EMA (No Cost-aware AL)
#     bars2 = ax.bar(x + width/2,
#                    df['EMA (No Cost-aware AL)'],
#                    width=width,
#                    color='black',  # Black color for No Cost-aware AL
#                    label='w/o Cost-aware AL',
#                    alpha=1.0)  # Add some transparency
    
#     # Add value labels on top of each bar
#     def autolabel(bars):
#         for bar in bars:
#             height = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width()/2., height - 0.05,
#                    f'{height:.1f}',
#                    ha='center', va='bottom', fontsize=12)
    
#     autolabel(bars1)
#     autolabel(bars2)
    
#     # Customize labels and ticks
#     ax.set_ylabel('Cost Reduction (%)', fontsize=font_size)
#     ax.set_xticks(x)
#     ax.set_xticklabels(df['Task Name'], rotation=0, ha='center', fontsize=font_size)  # Set labels to horizontal
#     ax.set_yticks([0, 15, 30, 45, 60])
#     plt.ylim(-0.1, 70)
    
#     # Customize tick parameters
#     ax.tick_params(axis='both', which='major', direction='in', length=6)
    
#     # Remove top and right spines
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     ax.yaxis.set_major_locator(FixedLocator([0, 15, 30, 45, 60, 75]))
#     ax.yaxis.set_major_formatter(FixedFormatter(['0', '15', '30', '45', '60', '75']))

#     # Add gridlines
#     plt.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.5)
    
#     # Add legend to the upper right corner, horizontally
#     ax.legend(fontsize=12, frameon=True, loc='upper right', bbox_to_anchor=(1, 1), ncol=1)
    
#     # Tight layout for better spacing
#     plt.tight_layout()
    
#     # Save plot
#     plt.savefig('cost_reduction.pdf', format='pdf', dpi=300,
#                 bbox_inches='tight', pad_inches=0.0)
#     plt.savefig('cost_reduction.png', format='png', dpi=300,
#                 bbox_inches='tight', pad_inches=0.0)
    
#     # Show the plot
#     plt.show()

# # Call the function to generate and display the bar chart
# plot_cost_reduction()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter

def plot_cost_reduction():
    # Data with percentages (multiply by 100)
    data = {
        'Task Name': ['Flux', 'MimicNet'],
        'EMA (Cost-aware AL)': [(1 - 0.3933) * 100, (1 - 0.8931575027) * 100],
        'EMA (No Cost-aware AL)': [(1 - 0.5096) * 100, (1 - 0.9251993005) * 100],
    }
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(data)
    
    # Figure and axis setup
    plt.figure(figsize=(4, 3.2), dpi=300)
    ax = plt.gca()
    
    # Font size
    font_size = 15
    
    # X locations for the groups
    x = np.arange(len(df['Task Name']))
    width = 0.4  # Width of each bar
    
    # Plot bars: EMA (Cost-aware AL)
    bars1 = ax.bar(x - width/2,
                   df['EMA (Cost-aware AL)'],
                   width=width,
                   color='#d62728',  # Red color for Cost-aware AL
                   label='w/ Cost-aware AL',
                   alpha=1.0)  # Add some transparency
    
    # Plot bars: EMA (No Cost-aware AL)
    bars2 = ax.bar(x + width/2,
                   df['EMA (No Cost-aware AL)'],
                   width=width,
                   color='black',  # Black color for No Cost-aware AL
                   label='w/o Cost-aware AL',
                   alpha=1.0)  # Add some transparency
    
    # Add value labels on top of each bar
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height - 0.05,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=12)
    
    autolabel(bars1)
    autolabel(bars2)
    
    # Customize labels and ticks
    ax.set_ylabel('Cost Reduction (%)', fontsize=font_size + 2)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Task Name'], rotation=0, ha='center', fontsize=font_size)  # Set labels to horizontal
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    plt.ylim(-0.1, 80)

    # 把y轴刻度加大
    plt.yticks(fontsize=font_size + 1)
    
    # Customize tick parameters
    ax.tick_params(axis='both', which='major', direction='in', length=6)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.yaxis.set_major_locator(FixedLocator([0, 20, 40, 60, 80, 100]))
    ax.yaxis.set_major_formatter(FixedFormatter(['0', '20', '40', '60', '80', '100']))

    # Add gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.5)
    
    # Add legend to the upper right corner, horizontally
    ax.legend(fontsize=12, frameon=True, loc='upper right', bbox_to_anchor=(1, 1), ncol=1)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save plot
    plt.savefig('cost_reduction.pdf', format='pdf', dpi=300,
                bbox_inches='tight', pad_inches=0.0)
    plt.savefig('cost_reduction.png', format='png', dpi=300,
                bbox_inches='tight', pad_inches=0.0)
    
    # Show the plot
    plt.show()

# Call the function to generate and display the bar chart
plot_cost_reduction()