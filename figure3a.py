import matplotlib.pyplot as plt
import pandas as pd

def plot_cost_distribution():
    # Set style parameters
    # plt.rcParams['font.family'] = 'Times New Roman'
    
    # Data for the bar chart
    data = {
        'Work/Task Name': ['Job Completion Pred','WAN Engineering', 'DCN Traffic Sim.', 'Task Resource Pred.','Adaptive Bitrate'],
        'Data Cost': [95.920946, 1391.342516, 177340.1538, 0.298208055456663,2.9458255226077847],
        'Training Cost': [1, 16, 1538.024055, 0.805,11.382736]
    }
    
    # Create and process DataFrame
    df = pd.DataFrame(data)
    df['Total Cost'] = df['Data Cost'] + df['Training Cost']
    df['Data Cost Percentage'] = (df['Data Cost'] / df['Total Cost']) * 100
    df['Training Cost Percentage'] = (df['Training Cost'] / df['Total Cost']) * 100
    
    # Create figure and axes
    plt.figure(figsize=(5.5, 4.5))
    ax = plt.gca()
    
    # Set font size
    font_size = 16
    
    # Create stacked bar chart
    ax.bar(df['Work/Task Name'], 
           df['Data Cost Percentage'], 
           label='Data Cost', 
           color='black',
           width=0.6)
    
    ax.bar(df['Work/Task Name'], 
           df['Training Cost Percentage'], 
           bottom=df['Data Cost Percentage'], 
           label='Training Cost', 
           color='#d62728',
           width=0.6)
    
    # Customize axis labels and ticks
    plt.ylabel('Cost Distribution (%)', fontsize=font_size)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=40, ha='right', fontsize=font_size + 1)
    plt.yticks([0, 25, 50, 75, 100], fontsize=font_size)
    
    # Customize tick parameters
    ax.tick_params(axis='both', which='major', 
                  direction='in', length=6)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    plt.legend(fontsize=13, frameon=True, loc='lower left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig('cost_distribution.pdf', format='pdf', 
                dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.savefig('cost_distribution.png', format='png',
                dpi=300, bbox_inches='tight', pad_inches=0.0)
    
    # Show plot
    plt.show()

# Call the function
plot_cost_distribution()