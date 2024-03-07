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
from urllib.parse import quote

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['svg.fonttype'] = 'none'



# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件的目录
current_directory = os.path.dirname(current_file_path)

# 改变当前工作目录
os.chdir(current_directory)

# 创建一个宽高比为2:1的figure
fig = plt.figure(figsize=(10, 5))     
ax = fig.add_subplot(1, 1, 1)
# 设置axes的宽高比为3:2
ax.set_aspect(3/2)

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
# Download the database file
filename = "GeoRoc.db"
zip_file = "GeoRoc.zip"
url = quote("https://github.com/GeoPyTool/GeoRocData/raw/main/GeoRoc.zip")

if not os.path.exists(filename):
    if os.path.exists(zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # zip_ref.extractall('.')
            total_files = len(zip_ref.infolist())
            extracted_files = 0
            for file in zip_ref.infolist():
                zip_ref.extract(file)
                extracted_files += 1
                print(f'Unzip Progress: {extracted_files / total_files * 100:.2f}%')
            # os.remove(zip_file)
    else:
        urllib.request.urlretrieve(url, zip_file, reporthook=download_progress)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # zip_ref.extractall('.')
            total_files = len(zip_ref.infolist())
            extracted_files = 0
            for file in zip_ref.infolist():
                zip_ref.extract(file)
                extracted_files += 1
                print(f'Unzip Progress: {extracted_files / total_files * 100:.2f}%')

# Connect to the database
conn = sqlite3.connect(filename)

# Read the data from the Current_Data table
df = pd.read_sql_query("SELECT * FROM Current_Data", conn)
# 输出'Type'的取值个数
# print(df['Type'].value_counts())


def visualize_data(df, name):
    data  = (df[(df['ROCK_TYPE'] == name) ]['Type'].value_counts())
    data.to_csv(name+'_type_list.csv')

    # 设置黄金分割比例
    golden_ratio = 1.618

    # 设置图形的宽度和高度
    fig_width = 10
    fig_height = fig_width / golden_ratio

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # 对数据进行排序
    data_sorted = data.sort_values(ascending=False)


    # 计算样品数量的中位数
    median = data_sorted.median()

    # 使用坐标轴的方法在图上绘制一条水平线表示中位数
    ax.axhline(y=median, color='k', linestyle='--')

    # 使用坐标轴的方法创建一个垂直的柱状图
    colors = ['black' if x >= median else 'gray' for x in data_sorted.values]
    bars = ax.bar(data_sorted.index, data_sorted.values, color=colors,alpha = 0.3)
    # 在每个柱条的内部最高处添加对应的数值
    for bar in bars:
        height = bar.get_height()
        if height >= median:
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    '{:d}'.format(int(height)), ha='center', va='bottom', rotation=90)

    # 找到第一个低于中位数的值的索引
    first_below_median_index = next(i for i, x in enumerate(data_sorted.values) if x < median)
    # 获取对应的x值
    first_below_median_x = data_sorted.index[first_below_median_index]
    # 在中位线旁边添加中位数的值
    ax.text(first_below_median_x, median, 'Median: {:.2f}'.format(median), color='k', va='bottom')

    # 设置x轴标签的颜色
    for ticklabel, tickcolor in zip(ax.get_xticklabels(), colors):
        ticklabel.set_color(tickcolor)
    
    # 找到所有大于30的值的索引
    indices_gt_30 = [i for i, x in enumerate(data_sorted.values) if x > 30]

    # 在图的右上部分显示这些索引和对应的值，分成三列
    for idx, i in enumerate(indices_gt_30):
        color = 'black' if data_sorted.values[i] >= median else 'gray'
        if idx % 3 == 0:  # 对于模数为0的索引，放在左列
            ax.text(0.33, 1 - 0.05 * (idx // 3) - 0.1, ' {}: {}'.format(data_sorted.index[i], data_sorted.values[i]),
                    color=color, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
        elif idx % 3 == 1:  # 对于模数为1的索引，放在中列
            ax.text(0.66, 1 - 0.05 * (idx // 3) - 0.1, ' {}: {}'.format(data_sorted.index[i], data_sorted.values[i]),
                    color=color, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
        else:  # 对于模数为2的索引，放在右列
            ax.text(1, 1 - 0.05 * (idx // 3) - 0.1, ' {}: {}'.format(data_sorted.index[i], data_sorted.values[i]),
                    color=color, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    # 使用坐标轴的方法添加标题和标签
    ax.set_title('Data Count of '+name+' Type')
    # ax.set_xlabel('Type')
    ax.set_ylabel('Number of Data Entries')
    # 设置x轴的右侧限制
    ax.set_xlim(right=data_sorted.index[-1])
    # 旋转x轴y轴的标签
    plt.xticks(rotation=45, ha='right',fontsize=9)
    plt.yticks(rotation=90)
    # 去掉图框的上方和右方的框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 自动调整图形
    fig.tight_layout()
    # 保存图形为SVG和PNG文件
    fig.savefig(name+'_data_stats.svg')
    fig.savefig(name+'_data_stats.jpg',dpi=600)

visualize_data(df, name='VOL')
visualize_data(df, name='PLU')

conn.close()