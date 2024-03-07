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

# Create a figure with aspect ratio 2:1
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1)

# Set the aspect ratio of axes to 3:2
ax.set_aspect(3/2)

start_time = time.time()


def download_progress(count, block_size, total_size):
    """
    This function is used to track the progress of a download.
    """
    global start_time
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    print(f"Downloaded {progress_size} of {total_size} bytes ({percent}% done), speed {speed} KB/s")
    if progress_size >= total_size:  # reset start_time for next download
        start_time = time.time()


def download_and_extract(filename, zip_file, url, extract_dir):
    """
    This function downloads a zip file from a given URL, extracts it, and opens the directory in the file explorer.

    Args:
    filename (str): The name of the file to check if it exists.
    zip_file (str): The name of the zip file to download.
    url (str): The URL to download the zip file from.
    extract_dir (str): The directory to extract the zip file to.
    """
    if not os.path.exists(filename):
        if os.path.exists(zip_file):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                total_files = len(zip_ref.infolist())
                extracted_files = 0
                for file in zip_ref.infolist():
                    zip_ref.extract(file, path=extract_dir)
                    extracted_files += 1
                    print(f'Unzip Progress: {extracted_files / total_files * 100:.2f}%')
        else:
            urllib.request.urlretrieve(url, zip_file, reporthook=download_progress)
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                total_files = len(zip_ref.infolist())
                extracted_files = 0
                for file in zip_ref.infolist():
                    zip_ref.extract(file, path=extract_dir)
                    extracted_files += 1
                    print(f'Unzip Progress: {extracted_files / total_files * 100:.2f}%')

        print(f'{zip_file} has been downloaded and extracted to \n {os.path.abspath(extract_dir)}')

        # Open the directory in the file explorer
        if platform.system() == "Windows":
            os.startfile(os.path.abspath(extract_dir))
        elif platform.system() == "Darwin":
            os.system(f'open "{os.path.abspath(extract_dir)}"')
        else:
            os.system(f'xdg-open "{os.path.abspath(extract_dir)}"')


def visualize_data(df, name):
    """
    This function is used to visualize the data.
    """
    data = (df[(df["ROCK TYPE"] == name)]['Type'].value_counts())

    # Set the golden ratio
    golden_ratio = 1.618

    # Set the width and height of the figure
    fig_width = 10
    fig_height = fig_width / golden_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Sort the data
    data_sorted = data.sort_values(ascending=False)

    # Calculate the median of the sample quantity
    median = data_sorted.median()

    # Draw a horizontal line to represent the median using the axes method
    ax.axhline(y=median, color='k', linestyle='--')

    # Create a vertical bar chart using the axes method
    colors = ['black' if x >= median else 'gray' for x in data_sorted.values]
    bars = ax.bar(data_sorted.index, data_sorted.values, color=colors, alpha=0.3)

    # Add the corresponding value at the highest point inside each bar
    for bar in bars:
        height = bar.get_height()
        if height >= median:
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    '{:d}'.format(int(height)), ha='center', va='bottom', rotation=90)

    # Find the index of the first value below the median
    first_below_median_index = next(i for i, x in enumerate(data_sorted.values) if x < median)

    # Get the corresponding x value
    first_below_median_x = data_sorted.index[first_below_median_index]

    # Add the value of the median next to the median line
    ax.text(first_below_median_x, median, 'Median: {:.2f}'.format(median), color='k', va='bottom')

    # Set the color of the x-axis labels
    for ticklabel, tickcolor in zip(ax.get_xticklabels(), colors):
        ticklabel.set_color(tickcolor)

    # Find the indices of all values greater than 30
    indices_gt_30 = [i for i, x in enumerate(data_sorted.values) if x > 30]

    # Display these indices and their corresponding values in the upper right part of the graph, divided into three columns
    for idx, i in enumerate(indices_gt_30):
        color = 'black' if data_sorted.values[i] >= median else 'gray'
        if idx % 3 == 0:  # For indices with a modulus of 0, put them in the left column
            ax.text(0.33, 1 - 0.05 * (idx // 3) - 0.1, ' {}: {}'.format(data_sorted.index[i], data_sorted.values[i]),
                    color=color, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
        elif idx % 3 == 1:  # For indices with a modulus of 1, put them in the middle column
            ax.text(0.66, 1 - 0.05 * (idx // 3) - 0.1, ' {}: {}'.format(data_sorted.index[i], data_sorted.values[i]),
                    color=color, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
        else:  # For indices with a modulus of 2, put them in the right column
            ax.text(1, 1 - 0.05 * (idx // 3) - 0.1, ' {}: {}'.format(data_sorted.index[i], data_sorted.values[i]),
                    color=color, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    # Add title and labels using the axes method
    ax.set_title('Data Count of ' + name + ' Type')
    ax.set_ylabel('Number of Data Entries')

    # Set the right limit of the x-axis
    ax.set_xlim(right=data_sorted.index[-1])

    # Rotate the labels of the x-axis and y-axis
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=90)

    # Remove the top and right frames of the graph
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust the figure automatically
    fig.tight_layout()

    # Save the figure as SVG and PNG files
    output_dir = 'Db_Stats'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig.savefig(output_dir + '/' + name + '_Stats.svg')
    fig.savefig(output_dir + '/' + name + '_Stats.jpg', dpi=600)
    data.to_csv(output_dir + '/' + name + '_List.csv')


# Define the filename, zip file, URL, and extraction directory
filename = "GeoRoc.db"
zip_file = "GeoRoc.zip"
url = "https://github.com/GeoPyTool/GeoRocData/raw/main/GeoRoc.zip"
extract_dir = "./"

# Download and extract the file
download_and_extract(filename, zip_file, url, extract_dir)

# Connect to the database
conn = sqlite3.connect(filename)

# Read the data from the Current_Data table
df = pd.read_sql_query("SELECT * FROM Current_Data", conn)

# Visualize the data
visualize_data(df, name='VOL')
visualize_data(df, name='PLU')

# Close the connection to the database
conn.close()