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
current_file_name = os.path.abspath(__file__)

# 获取当前文件的目录
current_directory = os.path.dirname(current_file_name)

# 改变当前工作目录
os.chdir(current_directory)


def split_csv(file_path):

    # 拆分路径和文件名
    dir_path, file_name = os.path.split(file_path)

    print("Dir Path:", dir_path)
    print("Filename:", file_name)

    # 获取output_dir的上层目录
    parent_dir = os.path.dirname(dir_path)

    # 连接上层目录和'Splited'这个文件夹
    splited_data_dir = os.path.join(parent_dir, 'Splited_Data')
    splited_abbreviations_dir = os.path.join(parent_dir, 'Splited_Abbreviations')
    splited_references_dir = os.path.join(parent_dir, 'Splited_References')

    # 检查splited_dir是否存在，如果不存在，创建它
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

    # df_data = pd.read_csv(file_path, skiprows=start_line_main, nrows=end_line_main-start_line_main, encoding='ISO-8859-1', low_memory=False)
    # df_abbreviations = pd.read_csv(file_path, skiprows=start_line_abbreviations, nrows=end_line_abbreviations-start_line_abbreviations, encoding='ISO-8859-1', low_memory=False)
    # df_references = pd.read_csv(file_path, skiprows=start_line_references, nrows=end_line_references-start_line_references, encoding='ISO-8859-1', low_memory=False)


    # Write each DataFrame to a separate CSV file
    df_data.to_csv(splited_data_dir+'/'+file_name.replace('.csv','')+'_data.csv', index=False)
    df_abbreviations.to_csv(splited_abbreviations_dir+'/'+file_name.replace('.csv','')+ '_abbreviations.csv', index=False)
    df_references.to_csv(splited_references_dir+'/'+file_name.replace('.csv','')+ '_references.csv', index=False)


# 设定目标路径名称
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