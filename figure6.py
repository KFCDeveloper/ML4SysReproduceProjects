import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_and_overhead_DOTE():
    # Data
    sample_sizes = [10249 * x for x in [0.2, 0.4, 0.6, 0.8]]
    overhead = [0.763, 2.780, 3.898, 3.966]  # seconds
    accuracy = [0.837, 0.848, 0.850, 0.852]
    accuracy = [x * 100 for x in accuracy]
    x = np.arange(len(sample_sizes))  # [0, 1, 2, 3]
    
    # Create a figure and its axes
    fig, ax_left = plt.subplots(figsize=(4, 3), dpi=300)
    
    # Plot bar chart for accuracy on left y-axis
    width = 0.6
    bars = ax_left.bar(
        x, accuracy,
        width=width,
        color="black",
        label="Model Accuracy"
    )
    
    ax_left.set_ylabel("Accuracy (%)", fontsize=19)
    ax_left.set_ylim([0.0, 100.0])  # Adjust if needed
    ax_left.set_xticks(x)
    ax_left.set_xticklabels([f"{int(p/1000)}" for p in sample_sizes], fontsize=16)
    
    # Create a twin axis on the right for overhead
    ax_right = ax_left.twinx()
    ax_right.set_ylabel("Overhead (s)", fontsize=19)
    max_overhead = max(overhead)
    ax_right.set_ylim(0, max_overhead * 1.2)
    
    ax_left.set_yticks([0, 25, 50, 75, 100])
    ax_left.set_yticklabels([0, 25, 50, 75, 100], fontsize=18)
    
    # Plot overhead with a line + triangle markers on the right axis
    line = ax_right.plot(
        x, overhead,
        color="#d62728",
        marker="^",
        markersize=8,
        linewidth=2,
        label="Overhead"
    )
    ax_left.set_xlabel("TCA Sampling Size (×10³)", fontsize=20)
    
    # Remove top and right spines from left axis to mimic style
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["top"].set_visible(False)
    
    # Add grid on left axis (optional)
    ax_left.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add a legend to indicate what each marker represents
    fig.legend(
        loc="upper left",  # 图例定位在上方中央
        bbox_to_anchor=(0.24, 1.0),  # 调整位置，使图例位于图像的中上部分
        fontsize=12,
        frameon=False,  # 不显示边框
        framealpha=1.0,  # 设置框的透明度（1.0表示完全不透明）
        edgecolor="black",  # 设置边框颜色
        handles=[
            bars[0],  # Reference for bars (accuracy)
            line[0]   # Reference for line (overhead)
        ],
        labels=[
            "Accuracy", 
            "Overhead"  # 调整图例标签为想要的样式
        ],
        ncol=1  # 设置图例项水平排列，ncol=2表示两项并排
    )
    
    # Tight layout for better spacing
    plt.tight_layout()
    plt.savefig("overhead_DOTE.pdf")
    plt.savefig("overhead_DOTE.png")
    plt.show()


def plot_accuracy_and_overhead_FLUX():
    # Data
    sample_sizes = [44754 * x for x in [0.2, 0.4, 0.6, 0.8]]
    overhead = [9.3, 37.5, 49.2, 49.4]  # seconds
    accuracy = [0.682, 0.775, 0.785, 0.785]
    accuracy = [x * 100 for x in accuracy]
    x = np.arange(len(sample_sizes))  # [0, 1, 2, 3]
    
    # Create a figure and its axes
    fig, ax_left = plt.subplots(figsize=(4, 3), dpi=300)
    
    # Plot bar chart for accuracy on left y-axis
    width = 0.6
    bars = ax_left.bar(
        x, accuracy,
        width=width,
        color="black",
        label="Model Accuracy"
    )
    
    ax_left.set_ylabel("Accuracy (%)", fontsize=19)
    ax_left.set_ylim([0.0, 100.0])  # Adjust if needed
    ax_left.set_xticks(x)
    ax_left.set_xticklabels([f"{p / 10000:.1f}" for p in sample_sizes], fontsize=16)
    
    
    # Create a twin axis on the right for overhead
    ax_right = ax_left.twinx()
    ax_right.set_ylabel("Overhead (s)", fontsize=19)
    max_overhead = max(overhead)
    ax_right.set_ylim(0, max_overhead * 1.2)

    ax_left.set_yticks([0, 25, 50, 75, 100])
    ax_left.set_yticklabels([0, 25, 50, 75, 100], fontsize=18)    

    # Plot overhead with a line + triangle markers on the right axis
    line = ax_right.plot(
        x, overhead,
        color="#d62728",
        marker="^",
        markersize=8,
        linewidth=2,
        label="Overhead"
    )
    ax_left.set_xlabel("TCA Sampling Size (×10⁴)", fontsize=20)
    
    # Remove top and right spines from left axis to mimic style
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["top"].set_visible(False)
    
    # Add grid on left axis (optional)
    ax_left.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add a legend to indicate what each marker represents
    fig.legend(
        loc="upper left",  # 图例定位在上方中央
        bbox_to_anchor=(0.24, 1.0),  # 调整位置，使图例位于图像的中上部分
        fontsize=12,
        frameon=False,  # 不显示边框
        framealpha=1.0,  # 设置框的透明度（1.0表示完全不透明）
        edgecolor="black",  # 设置边框颜色
        handles=[
            bars[0],  # Reference for bars (accuracy)
            line[0]   # Reference for line (overhead)
        ],
        labels=[
            "Accuracy", 
            "Overhead"  # 调整图例标签为想要的样式
        ],
        ncol=1  # 设置图例项水平排列，ncol=2表示两项并排
    )
    
    # Tight layout for better spacing
    plt.tight_layout()
    plt.savefig("overhead_FLUX.pdf")
    plt.savefig("overhead_FLUX.png")
    plt.show()


if __name__ == "__main__":
    plot_accuracy_and_overhead_DOTE()
    plot_accuracy_and_overhead_FLUX()