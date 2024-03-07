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

# Funtion to split the CSV file into three separate files
def split_csv(file_path):

    # Split the path and filename
    dir_path, file_name = os.path.split(file_path)

    print("Dir Path:", dir_path)
    print("Filename:", file_name)

    # Get the parent directory of output_dir
    parent_dir = os.path.dirname(dir_path)

    # Join the parent directory and the 'Splited' folder
    splited_data_dir = os.path.join(parent_dir, 'Splited_Data')
    splited_abbreviations_dir = os.path.join(parent_dir, 'Splited_Abbreviations')
    splited_references_dir = os.path.join(parent_dir, 'Splited_References')

    # Check if splited_dir exists, if not, create it
    for i in [splited_data_dir, splited_abbreviations_dir, splited_references_dir]:
        if not os.path.exists(i):
            os.makedirs(i)

    # Open the file and read the lines
    with open(file_path, 'r', encoding='ISO-8859-1') as f:
        lines = f.readlines()

    # Find the start and end lines for each section
    start_line_main = 0
    end_line_main = next(i for i, line in enumerate(lines) if 'Abbreviations:' in line) - 2
    start_line_abbreviations = end_line_main + 2
    end_line_abbreviations = next(i for i, line in enumerate(lines) if 'References:' in line) - 2
    start_line_references = end_line_abbreviations + 2
    end_line_references = len(lines) - 1

    # Read each section into a DataFrame    
    df_data = pd.read_csv(file_path, skiprows=start_line_main, nrows=end_line_main-start_line_main, encoding='ISO-8859-1', engine='python', on_bad_lines='warn')
    df_abbreviations = pd.read_csv(file_path, skiprows=start_line_abbreviations, nrows=end_line_abbreviations-start_line_abbreviations, encoding='ISO-8859-1', engine='python', on_bad_lines='warn')
    df_references = pd.read_csv(file_path, skiprows=start_line_references, nrows=end_line_references-start_line_references, encoding='ISO-8859-1', engine='python', on_bad_lines='warn')

    # Write each DataFrame to a separate CSV file
    df_data.to_csv(splited_data_dir+'/'+file_name.replace('.csv','')+'_data.csv', index=False)
    df_abbreviations.to_csv(splited_abbreviations_dir+'/'+file_name.replace('.csv','')+ '_abbreviations.csv', index=False)
    df_references.to_csv(splited_references_dir+'/'+file_name.replace('.csv','')+ '_references.csv', index=False)

    # Open the directory in the file explorer
    if platform.system() == "Windows":
        os.startfile(os.path.abspath(splited_data_dir))
    elif platform.system() == "Darwin":
        os.system(f'open "{os.path.abspath(splited_data_dir)}"')
    else:
        os.system(f'xdg-open "{os.path.abspath(splited_data_dir)}"')

# Set the target directory name
target_dir = 'GEOROC Compilation Rock Types'

# Get all files in target_dir
files = os.listdir(target_dir)

# For each file in target_dir, run split_csv
for file in files:
    file_name = os.path.join(target_dir, file)
    # Get the file extension
    _, extension = os.path.splitext(file_name)
    # If the file is a CSV file, run split_csv
    if extension.lower() == '.csv':
        split_csv(file_name)