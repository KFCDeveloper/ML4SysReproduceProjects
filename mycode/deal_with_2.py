import pandas as pd
# file_path = '/mydata/ns.py/2.txt'
data = pd.read_csv('/mydata/ns.py/2.csv')
grouped_data = data.groupby(data.columns[-1])
num_groups = len(grouped_data)
print("Number of groups:", num_groups)