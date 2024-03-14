import configparser
import csv
import json
import math
import os
import pickle
import platform
import sqlite3
import time
import itertools
import math

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.patches import ConnectionStyle
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.patches import ConnectionStyle, Polygon
from matplotlib.collections import PatchCollection
from matplotlib import collections


try:
    from importlib import metadata as importlib_metadata
except ImportError:
    # Backwards compatibility - importlib.metadata was added in Python 3.8
    import importlib_metadata

from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtWidgets import QAbstractItemView, QMainWindow, QApplication, QMenu, QToolBar, QFileDialog, QTableView, QVBoxLayout, QHBoxLayout, QWidget, QSlider,  QGroupBox , QLabel , QWidgetAction, QPushButton, QSizePolicy
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QVariantAnimation, Qt


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from PySide6.QtGui import QGuiApplication

from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] =  'truetype'


def Remove_LOI(filename = 'GeoRoc.db',output_dir='Corrected'):

    result_list = [['Label','Probability']]
    

    # Suppose you want a 2x2 grid of subplots
    # fig, ax_list = plt.subplots(2, 2, figsize=(10, 10), sharex=False, sharey=False)

    # Record the start time
    begin_time = time.time()

    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)

    # 获取当前文件的目录
    current_directory = os.path.dirname(current_file_path)
    # 改变当前工作目录
    os.chdir(current_directory)

    # 连接到数据库
    conn_in = sqlite3.connect(filename)

    # Read the data from the Corrected_Data table
    df = pd.read_sql_query("SELECT * FROM Current_Data", conn_in)

    # 获取所有包含"WT"的列 但去除 LOI(WT%)
    Major_Elements_List = [col for col in df.columns if 'WT' in col and 'LOI' not in col]
    PPM_Minor_Elements_List = [col for col in df.columns if 'PPM' in col]
    PPT_Minor_Elements_List = [col for col in df.columns if 'PPT' in col]

    Calculate_List = Major_Elements_List + PPM_Minor_Elements_List + PPT_Minor_Elements_List

    # 合并所有的列名
    all_columns = Calculate_List + ["Type", "TECTONIC SETTING"]

    other_columns = [col for col in df.columns if col not in all_columns]



    # 选择所有的列
    selected_df = df[all_columns]


    # 遍历所有列
    for col in (Major_Elements_List + PPM_Minor_Elements_List + PPT_Minor_Elements_List):
        # 如果列名包含"WT"，将该列的值转换为小数
        if 'WT' in col:
            selected_df[col] = selected_df[col]/ 100            
        # 如果列名包含"PPM"，将该列的值转换为小数
        elif 'PPM' in col:
            selected_df[col] = selected_df[col]/ 1000000
        # 如果列名包含"PPT"，将该列的值转换为小数
        elif 'PPT' in col:
            selected_df[col] = selected_df[col]/ 1000000000



    # conditions = (selected_df["Type"] == Type)
    # # for element in Elements_List:
    # #     conditions = conditions & (selected_df[element] ! = 0)
    # tag_df = selected_df[conditions]
            

    # # 找出所有的字符串类型列
    # str_columns = selected_df.select_dtypes(include=['object']).columns

    # # 打印出所有的字符串类型列
    # print(str_columns,selected_df[str_columns])

    
    # # 创建新的列"RAW_SUM"，其值为所有转换后的列的和
    # selected_df['RAW_SUM'] = selected_df[Calculate_List].sum(axis=1)
    # 'FE2O3(WT%)','FEO(WT%)','FEOT(WT%)'
    # 这里求和得到“RAW_SUM"的部分做一个筛选，其中除了'FE2O3(WT%)','FEO(WT%)','FEOT(WT%)'外，先加起来其他所有列；对于'FE2O3(WT%)','FEO(WT%)','FEOT(WT%)'这三列，如果'FEOT(WT%)'这一列不等于零就只用这一列加上其他的数据，而排除掉'FE2O3(WT%)','FEO(WT%)'；如果'FEOT(WT%)'等于零，就用'FE2O3(WT%)','FEO(WT%)'一起加进其他列的求和
    def calculate_sum(row):
        if row['FEOT(WT%)'] != 0:
            return row.drop(['FE2O3(WT%)', 'FEO(WT%)']).sum()
        else:
            return row.sum()

    selected_df['RAW_SUM'] = selected_df.apply(calculate_sum, axis=1)

    def normalize_row(row):
        divisor = row['RAW_SUM']
        if divisor == 0:
            divisor = 1
        return row[Calculate_List] / divisor

    selected_df[Calculate_List] = selected_df.apply(normalize_row, axis=1)

    
    selected_df['Corrected_SUM'] = selected_df[Calculate_List].sum(axis=1)


    # 遍历所有列
    for col in (Major_Elements_List + PPM_Minor_Elements_List + PPT_Minor_Elements_List):
        # 如果列名包含"WT"，将该列的值转换为百分数
        if 'WT' in col:
            selected_df[col] = selected_df[col]* 100
        # 如果列名包含"PPM"，将该列的值转换为PPM
        elif 'PPM' in col:
            selected_df[col] = selected_df[col]* 1000000
        # 如果列名包含"PPT"，将该列的值转换为PPT
        elif 'PPT' in col:
            selected_df[col] = selected_df[col]* 1000000000

    # print(selected_df['RAW_SUM'])
    print(selected_df)

    # def title_except_in_parentheses(s):
    #     parts = s.split('(')
    #     return parts[0].title() + '(' + parts[1].upper() if len(parts) > 1 else parts[0].title()

    # selected_df = selected_df.rename(columns=title_except_in_parentheses)

    selected_df[other_columns] = df[other_columns]
    # 将数据写入数据库
    
    # 连接到数据库
    conn_out= sqlite3.connect(output_dir+'/Remove_LOI_'+filename)
    selected_df.to_sql('Current_Data', conn_out, if_exists='replace', index=False)

    all_end_time = time.time()

    all_time_taken = all_end_time - begin_time

    
    conn_in.close()
    conn_out.close()
    print(f"All time taken: {all_time_taken:.3f} seconds")
    

# 文件名
filename = 'GeoRoc.db'
Type = 'Granite'
color_config_dir = 'Color_Config'
if not os.path.exists(color_config_dir):
    os.makedirs(color_config_dir)
output_dir = 'Corrected'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

Remove_LOI(filename, output_dir = 'Corrected')