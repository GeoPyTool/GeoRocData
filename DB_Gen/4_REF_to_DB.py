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
target_dir = 'Splited_References'
db_name = 'Ref.db'

def csv_to_db_with_check(target_dir,db_name):
    # 创建一个集合来存储已经读取的文件路径
    read_files = set()


    # 创建SQLite数据库
    conn = sqlite3.connect(db_name)
    # 打开记录文件，读取已经读取的文件路径
    with open(db_name.replace('.db','_')+'imported_files.txt', 'a+') as f:
        f.seek(0)
        read_files = set(line.strip() for line in f)

   

    # 遍历target_dir下的所有文件
    for file in os.listdir(target_dir):
        # 获取文件的完整路径
        file_path = os.path.join(target_dir, file)
        # 如果文件已经被读取，跳过
        if file_path in read_files:
            continue
        # 获取文件的扩展名
        _, extension = os.path.splitext(file_path)
        # 如果文件是CSV文件
        if extension.lower() == '.csv':
            try:
                data = pd.read_csv(file_path, encoding='utf-8', engine='python', on_bad_lines='warn', skiprows=1, header=None)
            except UnicodeDecodeError:
                try:
                    data = pd.read_csv(file_path, encoding='ISO-8859-1', engine='python', on_bad_lines='warn', skiprows=1, header=None)
                except UnicodeDecodeError:
                    data = pd.read_csv(file_path, encoding='Windows-1252', engine='python', on_bad_lines='warn', skiprows=1, header=None)

            data.columns = ['References']

            # Function to extract No. and Year from a string
            def extract_values(s):
                matches = re.findall(r'\[(\d+)\]', s)
                no = matches[0] if matches else None
                year = None
                for match in matches[1:]:
                    if match != no and int(match) >= 1800:
                        year = match
                        break
                return pd.Series([no, year])

            # Apply the function to each row of the 'data' DataFrame
            data[['Citation No.', 'Year']] = data.iloc[:, 0].apply(extract_values)
            # print(data.columns,data)

            # Check for duplicates
            print("Duplicates before removal: ", data.duplicated().sum())

            # Remove duplicates
            data = data.drop_duplicates()

            # Check for duplicates again
            print("Duplicates after removal: ", data.duplicated().sum())

            
            # 将数据写入SQLite数据库
            data.to_sql('References', conn, if_exists='append', index=False)
            with open(db_name.replace('.db','_')+'imported_files.txt', 'a') as f:
                f.write(file_path + '\n')

    # 提交事务并关闭连接
    conn.commit()
    conn.close()

csv_to_db_with_check(target_dir,db_name)