import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
import re
import os
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


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['svg.fonttype'] = 'none'


# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件的目录
current_directory = os.path.dirname(current_file_path)

# 改变当前工作目录
os.chdir(current_directory)

# 设定目标路径名称
target_dir = 'Splited_Data'

def add_filename_as_column(file_path):    

    # 拆分路径和文件名
    dir_path, file_name = os.path.split(file_path)

    print("Dir Path:", dir_path)
    print("Filename:", file_name)

    # 获取output_dir的上层目录
    parent_dir = os.path.dirname(dir_path)

    # 连接上层目录和'Type_Added'这个文件夹
    output_dir = os.path.join(parent_dir, 'Type_Added')

    # 检查output_dir是否存在，如果不存在，创建它
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

    encoding='utf-8'
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

        
# 设定目标路径名称
target_dir = 'Splited_Data'

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