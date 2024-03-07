import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
import re
import os       
import platform
import pandas as pd
import statistics
from fitter import Fitter
import seaborn as sns
from scipy.stats import norm, gaussian_kde
from scipy.interpolate import make_interp_spline, BSpline
from scipy.stats import norm, lognorm, gamma
import sqlite3

import time
import urllib.request
import zipfile
import toga, os, math, platform, toga_chart, json
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import scipy as sp
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.path import Path
from matplotlib.font_manager import FontProperties
from urllib.parse import quote

# Set the font for plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['svg.fonttype'] = 'none'

# Get the absolute path of the current file
current_file_name = os.path.abspath(__file__)

# Get the directory of the current file
current_directory = os.path.dirname(current_file_name)

# Change the current working directory
os.chdir(current_directory)

def generate_harker_diagram(filename, rock_type, major_oxides,output_dir):
    # Connect to the database
    conn = sqlite3.connect(filename)

    # Read the data from the Current_Data table
    df = pd.read_sql_query("SELECT * FROM Current_Data", conn)

    # Select columns
    selected_columns = df[["Type", "SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)", "FEOT(WT%)",  "CAO(WT%)", 
                    "MGO(WT%)", "MNO(WT%)", "K2O(WT%)", "NA2O(WT%)", "P2O5(WT%)","ROCK TYPE"]]

    # Filter rows where "ROCK TYPE" is rock_type and "SIO2(WT%)" is not 0
    tag_df = selected_columns[(selected_columns["ROCK TYPE"] == rock_type) & (selected_columns["SIO2(WT%)"] != 0)]

    # Output the count of "Type" values
    print(tag_df["Type"].value_counts())
    type_counts = tag_df["Type"].value_counts()
    type_counts.to_csv(output_dir + '/'+'Harker_'+rock_type+'.csv')

    # Create bi-variant plots
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    # Check if tag_color_dict.json file exists
    # 检查是否存在tag_color_dict.json文件
    if os.path.exists('Color_Config/'+rock_type+'_color_dict.json'):
        # 如果存在，从文件中读取tag_color_dict
        with open('Color_Config/'+rock_type+'_color_dict.json', 'r') as f:
            tag_color_dict = json.load(f)
    else:
        # 如果不存在，创建新的tag_color_dict并保存到文件中
        type_set = set(tag_df['Type'].unique())
        cmap = cm.get_cmap('rainbow', len(type_set))
        tag_color_dict = {type: cmap(i) for i, type in enumerate(type_set)}
        with open('Color_Config/'+rock_type+'_color_dict.json', 'w') as f:
            json.dump(tag_color_dict, f) 


    # Draw Harker scatter plots
    for i, oxide in enumerate(major_oxides):
        # Determine position on the subplot grid
        row = i // 3
        col = i % 3

        # Create scatter plot
        grouped = tag_df.groupby('Type')
        for label, group in grouped:
            axes[row, col].scatter(group ['SIO2(WT%)'], group [oxide], alpha=0.15, label=label, color=tag_color_dict[label], edgecolors='none')

        newoxide =  oxide.replace('(WT%)',' wt%')
        axes[row, col].set_xlabel('SiO2 wt%', fontsize=7)
        axes[row, col].set_ylabel(f'{newoxide}', fontsize=7)
        ax = axes[row, col]

        # Calculate the 1% and 9999% quantiles of the y-axis data
        q1 = tag_df[oxide].quantile(0.01)
        q3 = tag_df[oxide].quantile(0.9999)
        # Calculate the interquartile range
        iqr = q3 - q1
        # Calculate the most concentrated range
        lower_bound = min(tag_df[oxide])
        upper_bound = q3 + 0.1 * iqr

        # Limit the y-axis to the most concentrated range
        if not np.isnan(lower_bound) and not np.isnan(upper_bound) and not np.isinf(lower_bound) and not np.isinf(upper_bound):
            axes[row, col].set_ylim(lower_bound, upper_bound)
        else:
            print(f"Invalid y-axis limits for row {row}, column {col}: lower_bound={lower_bound}, upper_bound={upper_bound}")

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Calculate the number of data points within the view range
        visible_points = tag_df[(tag_df['SIO2(WT%)'] >= xlim[0]) & (tag_df['SIO2(WT%)'] <= xlim[1]) & 
                                (tag_df[oxide] >= ylim[0]) & (tag_df[oxide] <= ylim[1])]

        num_visible_points = len(visible_points)

        # Display the number of visible data points on the plot
        ax.text(0.05, 0.95, f'Visible points: {num_visible_points}', transform=ax.transAxes, verticalalignment='top')

    fig.tight_layout()

    # Create a legend on the right side of the whole figure
    legend = fig.legend(loc='center left', fontsize=4, bbox_to_anchor=(1, 0.5))
    # Save the plot, including the legend
    plt.savefig(output_dir + '/'+'Harker_'+rock_type+'.svg')
    plt.savefig(output_dir + '/'+'Harker_'+rock_type+'.png', dpi=600)

    conn.close()


# Set the parameters
filename = 'GeoRoc.db'
rock_type = 'VOL'
color_config_dir = 'Color_Config'
if not os.path.exists(color_config_dir):
    os.makedirs(color_config_dir)
output_dir = 'Harker'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# List of major oxides for y-axis
major_oxides = ["TIO2(WT%)", "AL2O3(WT%)", "FEOT(WT%)",  "CAO(WT%)", 
                "MGO(WT%)", "MNO(WT%)", "K2O(WT%)", "NA2O(WT%)", "P2O5(WT%)"]
# generate_harker_diagram(filename, rock_type, major_oxides)
generate_harker_diagram(filename,'VOL', major_oxides,output_dir)
generate_harker_diagram(filename,'PLU', major_oxides,output_dir)