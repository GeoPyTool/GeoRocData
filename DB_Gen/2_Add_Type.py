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

# Set the target directory name
target_dir = 'Splited_Data'

# Add the filename as a new column to the CSV file
def add_filename_as_column(file_path):    
    """
    This function adds the filename as a new column to the CSV file.
    """

    # Split the file path into directory path and file name
    dir_path, file_name = os.path.split(file_path)

    print("Directory Path:", dir_path)
    print("Filename:", file_name)

    # Get the parent directory of the output_dir
    parent_dir = os.path.dirname(dir_path)

    # Join the parent directory with the 'Type_Added' folder
    output_dir = os.path.join(parent_dir, 'Type_Added')

    # Check if output_dir exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a new file path for the modified data
    output_path = os.path.join(output_dir, file_name.replace('.csv', '') + '_type_added.csv')

    # Process the filename
    split_file = file_name.split('_')
    if len(split_file) > 1:
        processed_file = split_file[1].capitalize()
    else:
        pass  # Skip this file if it doesn't contain an underscore

    # Remove '.csv' from the processed filename
    processed_file = processed_file.replace('.csv', '')

    # Try to read the file with different encodings
    try:
        data = pd.read_csv(file_path, encoding='utf-8', engine='python', on_bad_lines='warn')
        encoding='utf-8'
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(file_path, encoding='ISO-8859-1', engine='python', on_bad_lines='warn')
            encoding='ISO-8859-1'
        except UnicodeDecodeError:
            data = pd.read_csv(file_path, encoding='Windows-1252', engine='python', on_bad_lines='warn')
            encoding='Windows-1252'

    # Check if the first column is already 'Type'
    if data.columns[0] != 'Type':
        # Add 'Type' to the first column
        data.insert(0, 'Type', processed_file)

    # Write the data back to a new file
    data.to_csv(output_path, index=False, encoding=encoding)

# Get all files in target_dir
files = os.listdir(target_dir)

# For each file in target_dir, run split_csv
for file in files:
    file_name = os.path.join(target_dir, file)
    # Get the file extension
    _, extension = os.path.splitext(file_name)
    # If the file is a CSV file, run split_csv
    if extension.lower() == '.csv':
        add_filename_as_column(file_name)