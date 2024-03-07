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

# Set the target directory and database names
target_dir = 'Type_Added'
db_name = 'GeoRoc.db'

def csv_to_db_with_check(target_dir, db_name):
    """
    This function reads CSV files from the target directory and writes them to a SQLite database.
    It also checks for and removes duplicate rows in the CSV files.
    """

    # Create a set to store the paths of files that have been read
    read_files = set()

    # Create a SQLite database
    conn = sqlite3.connect(db_name)

    # Open the record file and read the paths of files that have been read
    with open(db_name.replace('.db', '_') + 'imported_files.txt', 'a+') as f:
        f.seek(0)
        read_files = set(line.strip() for line in f)

    # Iterate over all files in the target directory
    for file in os.listdir(target_dir):
        # Get the full path of the file
        file_path = os.path.join(target_dir, file)

        # If the file has been read, skip it
        if file_path in read_files:
            continue

        # Get the extension of the file
        _, extension = os.path.splitext(file_path)

        # If the file is a CSV file
        if extension.lower() == '.csv':
            # Try to read the file with different encodings
            try:
                data = pd.read_csv(file_path, encoding='utf-8', engine='python', on_bad_lines='warn')
            except UnicodeDecodeError:
                try:
                    data = pd.read_csv(file_path, encoding='ISO-8859-1', engine='python', on_bad_lines='warn')
                except UnicodeDecodeError:
                    data = pd.read_csv(file_path, encoding='Windows-1252', engine='python', on_bad_lines='warn')

            # Check for duplicates
            print("Duplicates before removal: ", data.duplicated().sum())

            # Remove duplicates
            data = data.drop_duplicates()

            # Check for duplicates again
            print("Duplicates after removal: ", data.duplicated().sum())

            # Write the data to the SQLite database
            data.to_sql('Current_Data', conn, if_exists='append', index=False)

            # Record the path of the file that has been read
            with open(db_name.replace('.db', '_') + 'imported_files.txt', 'a') as f:
                f.write(file_path + '\n')

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()

# Call the function
csv_to_db_with_check(target_dir, db_name)