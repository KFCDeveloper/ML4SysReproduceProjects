import os
import csv
import re

# Define the path to the folder containing the log files
log_folder_path = "firm_scratch_results"

# Initialize a dictionary to store the cumulative moving averages and counts per iteration
iteration_sums = {}
iteration_counts = {}

# Regular expression to extract iteration and moving average
pattern = re.compile(r"Iteration: (\d+) .* Moving average: ([\d\.]+)")

# Iterate through all files in the folder
for log_file in os.listdir(log_folder_path):
    if log_file.endswith(".log"):
        log_file_path = os.path.join(log_folder_path, log_file)
        
        # Open and read the log file
        with open(log_file_path, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    iteration = int(match.group(1))
                    moving_average = float(match.group(2))

                    # Update the sums and counts for the iteration
                    if iteration not in iteration_sums:
                        iteration_sums[iteration] = 0
                        iteration_counts[iteration] = 0
                    iteration_sums[iteration] += moving_average
                    iteration_counts[iteration] += 1

# Compute the average moving averages for each iteration
average_moving_averages = {
    iteration: iteration_sums[iteration] / iteration_counts[iteration]
    for iteration in iteration_sums
}

# Write the results into a CSV file
output_csv_path = os.path.join(log_folder_path, "average_moving_averages.csv")
with open(output_csv_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["iteration", "average"])

    for iteration in sorted(average_moving_averages.keys()):
        csv_writer.writerow([iteration, average_moving_averages[iteration]])

print(f"Average moving averages written to {output_csv_path}")
