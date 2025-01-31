
# import matplotlib.pyplot as plt
# import numpy as np

# def plot_two_groups_with_tasks():
#     # Pearson values
#     l2_values = [0.02, -0.23, -0.40]
#     mmd_values = [0.35, 0.38, 0.55]

#     # Task labels
#     tasks = ["Flux", "DOTE", "Firm"]

#     # Colors for each task
#     task_colors = ["#d62728", "#1f77b4", "black"]

#     # X positions for each group
#     x_l2 = [0, 1, 2]
#     x_mmd = [4, 5, 6]

#     # Create the figure and axes
#     plt.figure(figsize=(6, 4), dpi=300)
#     ax = plt.gca()

#     # Plot bars for L2
#     bars_l2 = ax.bar(
#         x_l2,
#         l2_values,
#         color=task_colors,
#         alpha=1.0,
#         width=0.8
#     )

#     # Plot bars for MMD
#     bars_mmd = ax.bar(
#         x_mmd,
#         mmd_values,
#         color=task_colors,
#         alpha=1.0,
#         width=0.8
#     )

#     # Annotate the bars with values
#     for bar_group, vals in [(bars_l2, l2_values), (bars_mmd, mmd_values)]:
#         for bar, coef in zip(bar_group, vals):
#             height = bar.get_height()
#             x_pos = bar.get_x() + bar.get_width() / 2
#             y_offset = 0.05 if height >= 0 else -0.08
#             plt.text(
#                 x_pos,
#                 height + y_offset,
#                 f"{coef:.2f}",
#                 ha='center',
#                 va='bottom' if height >= 0 else 'top',
#                 fontsize=14
#             )

#     # Set the x-axis at y=0
#     ax.spines['bottom'].set_position(('data', 0))
#     ax.spines['bottom'].set_visible(True)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(True)

#     # Make the y-axis and its labels thicker
#     ax.spines['left'].set_linewidth(2)
#     ax.spines['bottom'].set_linewidth(2)
#     ax.tick_params(axis='y', width=2, labelsize=16)
#     ax.tick_params(axis='x', bottom=False)

#     # Add y-axis label with increased font size
#     ax.set_ylabel("Pearson(Sim, Acc)", fontsize=18)

#     # Adjust y-axis tick label size and sparsity
#     ax.set_yticks(np.arange(-1, 1.5, 0.5))

#     # Remove default x-axis ticks and labels
#     ax.set_xticks([])

#     # Add custom x-axis labels for L2 and MMD
#     plt.text(1, -0.05, "", ha='center', va='top', fontsize=15)
#     plt.text(5, -0.05, "", ha='center', va='top', fontsize=15)

#     # Draw a horizontal line at y=0
#     ax.axhline(0, color='black', linewidth=0.8)

#     # Draw a vertical dashed line to split L2 and MMD sections
#     ax.axvline(3, color='black', linestyle='--', linewidth=1.5)

#     # Optional: Adjust y-limits for better visualization
#     ax.set_ylim(-1.1, 1.1)

#     # Add a legend block for tasks
#     legend_handles = []
#     for t_label, color in zip(tasks, task_colors):
#         legend_handles.append(
#             plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor='none', label=t_label)
#         )
#     ax.legend(handles=legend_handles, fontsize=15, loc='upper left', bbox_to_anchor=(0.0, 0.9))

#     # Add "Higher is better" annotation with an upward arrow, below the x-axis
#     plt.annotate(
#         "Higher\nis better",
#         xy=(5.2, -0.1),
#         xytext=(5.2, -0.7),
#         fontsize=20,
#         ha='center',
#         va='center',
#         arrowprops=dict(facecolor='black', arrowstyle="->", lw=1.5)
#     )

#     # Add x-axis group labels at the top of the graph
#     plt.text(1, 1.0, "L2", ha='center', va='center', fontsize=18)
#     plt.text(5, 1.0, "MMD", ha='center', va='center', fontsize=18)

#     plt.tight_layout()
#     plt.savefig("similarity_Acc.pdf")
#     plt.savefig("similarity_Acc.png")
#     plt.show()

# # Call the function to create and display the plot
# plot_two_groups_with_tasks()


# import matplotlib.pyplot as plt
# import numpy as np

# def plot_two_groups_with_tasks_second():
#     # Pearson values
#     l2_values = [-0.301, -0.147, 0.071]
#     mmd_values = [0.757, 0.290, 0.500]

#     # Task labels
#     tasks = ["Flux", "DOTE", "FIRM"]

#     # Colors for each task
#     task_colors = ["#d62728", "#1f77b4", "black"]

#     # X positions for each group
#     x_l2 = [0, 1, 2]
#     x_mmd = [4, 5, 6]

#     # Create the figure and axes
#     plt.figure(figsize=(6, 4), dpi=300)
#     ax = plt.gca()

#     # Plot bars for L2
#     bars_l2 = ax.bar(
#         x_l2,
#         l2_values,
#         color=task_colors,
#         alpha=1.0,
#         width=0.8
#     )

#     # Plot bars for MMD
#     bars_mmd = ax.bar(
#         x_mmd,
#         mmd_values,
#         color=task_colors,
#         alpha=1.0,
#         width=0.8
#     )

#     # Annotate the bars with values
#     for bar_group, vals in [(bars_l2, l2_values), (bars_mmd, mmd_values)]:
#         for bar, coef in zip(bar_group, vals):
#             height = bar.get_height()
#             x_pos = bar.get_x() + bar.get_width() / 2
#             y_offset = 0.05 if height >= 0 else -0.08
#             plt.text(
#                 x_pos,
#                 height + y_offset,
#                 f"{coef:.2f}",
#                 ha='center',
#                 va='bottom' if height >= 0 else 'top',
#                 fontsize=14
#             )

#     # Set the x-axis at y=0
#     ax.spines['bottom'].set_position(('data', 0))
#     ax.spines['bottom'].set_visible(True)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(True)

#     # Make the y-axis and its labels thicker
#     ax.spines['left'].set_linewidth(2)
#     ax.spines['bottom'].set_linewidth(2)
#     ax.tick_params(axis='y', width=2, labelsize=16)
#     ax.tick_params(axis='x', bottom=False)

#     # Add y-axis label with increased font size
#     ax.set_ylabel("Pearson(Sim, Speedup)", fontsize=18)

#     # Adjust y-axis tick label size and sparsity
#     ax.set_yticks(np.arange(-1, 1.5, 0.5))

#     # Remove default x-axis ticks and labels
#     ax.set_xticks([])

#     # Add custom x-axis labels for L2 and MMD
#     plt.text(1, -0.05, "", ha='center', va='top', fontsize=15)
#     plt.text(5, -0.05, "", ha='center', va='top', fontsize=15)

#     # Draw a horizontal line at y=0
#     ax.axhline(0, color='black', linewidth=0.8)

#     # Draw a vertical dashed line to split L2 and MMD sections
#     ax.axvline(3, color='black', linestyle='--', linewidth=1.5)

#     # Optional: Adjust y-limits for better visualization
#     ax.set_ylim(-1.1, 1.1)

#     # Add a legend block for tasks
#     legend_handles = []
#     for t_label, color in zip(tasks, task_colors):
#         legend_handles.append(
#             plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor='none', label=t_label)
#         )
#     ax.legend(handles=legend_handles, fontsize=15, loc='upper left', bbox_to_anchor=(0.0, 0.9))

#     # Add "Higher is better" annotation with an upward arrow, below the x-axis
#     plt.annotate(
#         "Higher\nis better",
#         xy=(5.2, -0.1),
#         xytext=(5.2, -0.7),
#         fontsize=20,
#         ha='center',
#         va='center',
#         arrowprops=dict(facecolor='black', arrowstyle="->", lw=1.5)
#     )

#     # Add x-axis group labels at the top of the graph
#     plt.text(1, 1.0, "L2", ha='center', va='center', fontsize=18)
#     plt.text(5, 1.0, "MMD", ha='center', va='center', fontsize=18)

#     plt.tight_layout()
#     plt.savefig("similarity_speedup.pdf")
#     plt.savefig("similarity_speedup.png")
#     plt.show()

# # Call the function to create and display the plot
# plot_two_groups_with_tasks_second()

import matplotlib.pyplot as plt
import numpy as np

def plot_two_groups_with_tasks():
    # Pearson values
    l2_values = [0.02, -0.23, -0.40]
    mmd_values = [0.35, 0.38, 0.55]

    # Task labels
    tasks = ["Flux", "DOTE", "Firm"]

    # Colors for each task
    task_colors = ["#d62728", "#1f77b4", "black"]

    # X positions for each group
    x_l2 = [0, 1, 2]
    x_mmd = [4, 5, 6]

    # Create the figure and axes
    plt.figure(figsize=(6, 4), dpi=300)
    ax = plt.gca()

    # Plot bars for L2
    bars_l2 = ax.bar(
        x_l2,
        l2_values,
        color=task_colors,
        alpha=1.0,
        width=1.0  # Increased width
    )

    # Plot bars for MMD
    bars_mmd = ax.bar(
        x_mmd,
        mmd_values,
        color=task_colors,
        alpha=1.0,
        width=1.0  # Increased width
    )

    # Annotate the bars with values
    for bar_group, vals in [(bars_l2, l2_values), (bars_mmd, mmd_values)]:
        for bar, coef in zip(bar_group, vals):
            height = bar.get_height()
            x_pos = bar.get_x() + bar.get_width() / 2
            y_offset = 0.05 if height >= 0 else -0.08
            plt.text(
                x_pos,
                height + y_offset,
                f"{coef:.2f}",
                ha='center',
                va='bottom' if height >= 0 else 'top',
                fontsize=16  # Increased font size
            )

    # Set the x-axis at y=0
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)

    # Make the y-axis and its labels thicker
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(axis='y', width=2, labelsize=18)  # Increased label size
    ax.tick_params(axis='x', bottom=False)

    # Add y-axis label with increased font size
    ax.set_ylabel("Pearson(Sim, Acc)", fontsize=20)  # Increased font size

    # Adjust y-axis tick label size and sparsity
    ax.set_yticks(np.arange(-1, 1.5, 0.5))

    # Remove default x-axis ticks and labels
    ax.set_xticks([])

    # Add custom x-axis labels for L2 and MMD
    plt.text(1, -0.05, "", ha='center', va='top', fontsize=17)  # Increased font size
    plt.text(5, -0.05, "", ha='center', va='top', fontsize=17)  # Increased font size

    # Draw a horizontal line at y=0
    ax.axhline(0, color='black', linewidth=0.8)

    # Draw a vertical dashed line to split L2 and MMD sections
    ax.axvline(3, color='black', linestyle='--', linewidth=1.5)

    # Optional: Adjust y-limits for better visualization
    ax.set_ylim(-1.1, 1.1)

    # Add a legend block for tasks
    legend_handles = []
    for t_label, color in zip(tasks, task_colors):
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor='none', label=t_label)
        )
    ax.legend(handles=legend_handles, fontsize=15.5, loc='upper left', bbox_to_anchor=(0.0, 0.9))  # Increased font size

    # Add "Higher is better" annotation with an upward arrow, below the x-axis
    plt.annotate(
        "Higher\nis better",
        xy=(5.2, -0.1),
        xytext=(5.2, -0.7),
        fontsize=22,  # Increased font size
        ha='center',
        va='center',
        arrowprops=dict(facecolor='black', arrowstyle="->", lw=1.5)
    )

    # Add x-axis group labels at the top of the graph
    plt.text(1, 1.0, "L2", ha='center', va='center', fontsize=20)  # Increased font size
    plt.text(5, 1.0, "MMD", ha='center', va='center', fontsize=20)  # Increased font size

    plt.tight_layout()
    plt.savefig("similarity_Acc.pdf")
    plt.savefig("similarity_Acc.png")
    plt.show()

# Call the function to create and display the plot
plot_two_groups_with_tasks()


def plot_two_groups_with_tasks_second():
    # Pearson values
    l2_values = [-0.301, -0.147, 0.071]
    mmd_values = [0.757, 0.290, 0.500]

    # Task labels
    tasks = ["Flux", "DOTE", "FIRM"]

    # Colors for each task
    task_colors = ["#d62728", "#1f77b4", "black"]

    # X positions for each group
    x_l2 = [0, 1, 2]
    x_mmd = [4, 5, 6]

    # Create the figure and axes
    plt.figure(figsize=(6, 4), dpi=300)
    ax = plt.gca()

    # Plot bars for L2
    bars_l2 = ax.bar(
        x_l2,
        l2_values,
        color=task_colors,
        alpha=1.0,
        width=1.0  # Increased width
    )

    # Plot bars for MMD
    bars_mmd = ax.bar(
        x_mmd,
        mmd_values,
        color=task_colors,
        alpha=1.0,
        width=1.0  # Increased width
    )

    # Annotate the bars with values
    for bar_group, vals in [(bars_l2, l2_values), (bars_mmd, mmd_values)]:
        for bar, coef in zip(bar_group, vals):
            height = bar.get_height()
            x_pos = bar.get_x() + bar.get_width() / 2
            y_offset = 0.05 if height >= 0 else -0.08
            plt.text(
                x_pos,
                height + y_offset,
                f"{coef:.2f}",
                ha='center',
                va='bottom' if height >= 0 else 'top',
                fontsize=16  # Increased font size
            )

    # Set the x-axis at y=0
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)

    # Make the y-axis and its labels thicker
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(axis='y', width=2, labelsize=18)  # Increased label size
    ax.tick_params(axis='x', bottom=False)

    # Add y-axis label with increased font size
    ax.set_ylabel("Pearson(Sim, Speedup)", fontsize=20)  # Increased font size

    # Adjust y-axis tick label size and sparsity
    ax.set_yticks(np.arange(-1, 1.5, 0.5))

    # Remove default x-axis ticks and labels
    ax.set_xticks([])

    # Add custom x-axis labels for L2 and MMD
    plt.text(1, -0.05, "", ha='center', va='top', fontsize=17)  # Increased font size
    plt.text(5, -0.05, "", ha='center', va='top', fontsize=17)  # Increased font size

    # Draw a horizontal line at y=0
    ax.axhline(0, color='black', linewidth=0.8)

    # Draw a vertical dashed line to split L2 and MMD sections
    ax.axvline(3, color='black', linestyle='--', linewidth=1.5)

    # Optional: Adjust y-limits for better visualization
    ax.set_ylim(-1.1, 1.1)

    # Add a legend block for tasks
    legend_handles = []
    for t_label, color in zip(tasks, task_colors):
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor='none', label=t_label)
        )
    ax.legend(handles=legend_handles, fontsize=15.5, loc='upper left', bbox_to_anchor=(0.0, 0.9))  # Increased font size

    # Add "Higher is better" annotation with an upward arrow, below the x-axis
    plt.annotate(
        "Higher\nis better",
        xy=(5.2, -0.1),
        xytext=(5.2, -0.7),
        fontsize=22,  # Increased font size
        ha='center',
        va='center',
        arrowprops=dict(facecolor='black', arrowstyle="->", lw=1.5)
    )

    # Add x-axis group labels at the top of the graph
    plt.text(1, 1.0, "L2", ha='center', va='center', fontsize=20)  # Increased font size
    plt.text(5, 1.0, "MMD", ha='center', va='center', fontsize=20)  # Increased font size

    plt.tight_layout()
    plt.savefig("similarity_speedup.pdf")
    plt.savefig("similarity_speedup.png")
    plt.show()

# Call the function to create and display the plot
plot_two_groups_with_tasks_second()