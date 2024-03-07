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

# Function to monitor the download progress
start_time = time.time()
def download_progress(count, block_size, total_size):
    global start_time
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    print(f"Downloaded {progress_size} of {total_size} bytes ({percent}% done), speed {speed} KB/s")
    if progress_size >= total_size:  # reset start_time for next download
        start_time = time.time()

# Function to download and extract a zip file
def download_and_extract(target_dir, repo_link, date):
    zip_file = target_dir+".zip"
    url = quote(repo_link+"/raw/main/"+date + "/"+zip_file, safe='/:')

    if not os.path.exists(target_dir):
        if os.path.exists(zip_file):
            extract_dir = os.path.splitext(zip_file)[0]
            os.makedirs(extract_dir, exist_ok=True)

            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                total_files = len(zip_ref.infolist())
                extracted_files = 0
                for file in zip_ref.infolist():
                    zip_ref.extract(file, path=extract_dir)
                    extracted_files += 1
                    print(f'Unzip Progress: {extracted_files / total_files * 100:.2f}%')
        else:
            urllib.request.urlretrieve(url, zip_file, reporthook=download_progress)
            extract_dir = os.path.splitext(zip_file)[0]
            os.makedirs(extract_dir, exist_ok=True)

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

# Function to fix a line in a CSV file
def fix_line(line):
    pattern = r'(?<!")"(?!")'
    fixed_line = re.sub(pattern, '', line)
    return fixed_line

# Function to fix a CSV file
def fix_csv_file(file_path):
    with open(file_path, 'r', newline='', encoding='ISO-8859-1') as f:
        reader = csv.reader(f)
        rows = list(reader)

    has_extra_quotes = any('"' in field for row in rows for field in row)

    if not has_extra_quotes:
        return
    else:
        print(f'File "{file_path}" has isolated quotes error.')
        

    with open(file_path, 'w', newline='', encoding='ISO-8859-1') as f:
        writer = csv.writer(f)
        for row in rows:
            fixed_row = [fix_line(field) for field in row]
            writer.writerow(fixed_row)

# Download the data file    
target_dir = "GEOROC Compilation Rock Types"
repo_link = "https://github.com/GeoPyTool/GeoRocData"
date = "2023-12-01"
download_and_extract(target_dir, repo_link, date)

# Fix Quota Error in the downloaded CSV files
files = os.listdir(target_dir)
for file in files:
    file_name = os.path.join(target_dir, file)
    _, extension = os.path.splitext(file_name)
    if extension.lower() == '.csv':
        fix_csv_file(file_name)