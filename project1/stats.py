from ucimlrepo import fetch_ucirepo 
import seaborn as sns
import pandas as pd
import statistics
from prettytable import PrettyTable
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class StatsObject:
    def __init__(self, label):
        self.label = label
        self.count = 0
        self.sum = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.mean = None
        self.stdev = None
        self.median = None
        self.percentile_25 = None
        self.percentile_75 = None
        self.skewness = None
        self.kurtosis = None
        self.data = []
        
    def update(self, val):
        if val is not None:
            self.data.append(val)
            self.count += 1
            self.sum += val
            self.max = max(self.max, val)
            self.min = min(self.min, val)
            
def create_box_plots(data: list[StatsObject]):
    # Set the random seed for reproducibility
    np.random.seed(42)
    
    # Create a dictionary to hold the random data
    df_dict = {}
    for entry in data:
        df_dict[entry.label] = entry.data
        
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(df_dict)
    
    # Set up the matplotlib figure and axes
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    
    # Create box plots in a 3x3 grid
    for i, ax in enumerate(axes.flat):
        if i < df.shape[1]:  # Check to avoid IndexError if there are fewer than 9 entries
            sns.boxplot(y=df.iloc[:, i], ax=ax, width=0.3)
            ax.set_title(f'Box Plot for {df.columns[i]}')
            ax.set_ylabel('Values')
        else:
            ax.axis('off')  # Turn off empty subplots if there are less than 9
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig('boxplots.png')

# fetch dataset 
glass_identification = fetch_ucirepo(id=42) 
  
# data (as pandas dataframes) 
X = glass_identification.data.features 

RI_obj = StatsObject('RI')
Na_obj = StatsObject('Na')
Mg_obj = StatsObject('Mg')
Al_obj = StatsObject('Al')
Si_obj = StatsObject('Si')
K_obj = StatsObject('K')
Ca_obj = StatsObject('Ca')
Ba_obj = StatsObject('Ba')
Fe_obj = StatsObject('Fe')

stats_objects = [
    RI_obj, 
    Na_obj, 
    Mg_obj, 
    Al_obj, 
    Si_obj, 
    K_obj, 
    Ca_obj, 
    Ba_obj, 
    Fe_obj
]

for index, row in X.iterrows():
    for stats_obj in stats_objects:
        stats_obj.update(row[stats_obj.label])

# printing attribute statistics summary
table = PrettyTable()
table.field_names = ['Attribute', 'Count', 'Mean', 'Std. Dev', 'Min', '25th Percentile', 'Median', '75th Percentile', 'Max', 'Skewness', 'Kurtosis']

for stats_obj in stats_objects:
    stats_obj.mean = stats_obj.sum / stats_obj.count
    stats_obj.stdev = statistics.stdev(stats_obj.data)
    percentiles = statistics.quantiles(stats_obj.data, n=100)
    stats_obj.percentile_25 = percentiles[24]
    stats_obj.percentile_75 = percentiles[74]
    stats_obj.median = statistics.median(stats_obj.data)
    stats_obj.skewness = stats.skew(stats_obj.data)
    stats_obj.kurtosis = stats.kurtosis(stats_obj.data)
    row = [stats_obj.label, stats_obj.count, stats_obj.mean, stats_obj.stdev, stats_obj.min, stats_obj.percentile_25, stats_obj.median, stats_obj.percentile_75, stats_obj.max, stats_obj.skewness, stats_obj.kurtosis]
    table.add_row(row)
    
    # print_row = [stats_obj.label, stats_obj.count, round(stats_obj.mean, 3), round(stats_obj.stdev, 3), round(stats_obj.min, 3), round(stats_obj.percentile_25, 3), round(stats_obj.median, 3), round(stats_obj.percentile_75, 3), round(stats_obj.max, 3), round(stats_obj.skewness, 3), round(stats_obj.kurtosis, 3)]
    # print(f"{' & '.join(map(str, print_row))} \\\ ")

    
print(table)

create_box_plots(stats_objects)