# 导入所需的库
import pkg_resources
import types
import json  # 用于处理JSON数据
import pickle  # 用于序列化和反序列化Python对象结构
import sqlite3  # 用于SQLite数据库操作
import sys  # 提供对Python运行时环境的访问
import re  # 用于正则表达式
import os  # 提供了丰富的方法来处理文件和目录
import numpy as np  # 用于科学计算

# 导入matplotlib库的部分模块，用于绘图
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.patches import ConnectionStyle, Polygon
from matplotlib.collections import PatchCollection
from matplotlib import collections

# 导入importlib_metadata，用于处理Python包的元数据
try:
    from importlib import metadata as importlib_metadata
except ImportError:
    # Backwards compatibility - importlib.metadata was added in Python 3.8
    import importlib_metadata

# 导入PySide6库的部分模块，用于GUI编程
from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtWidgets import (QAbstractItemView, QMainWindow, QApplication, QMenu, QToolBar, QFileDialog, QTableView, QVBoxLayout, QHBoxLayout, QWidget, QSlider,  QGroupBox , QLabel , QWidgetAction, QPushButton, QSizePolicy)
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QVariantAnimation, Qt

# 导入matplotlib的Qt后端，用于在Qt应用程序中显示matplotlib图形
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 导入pandas，用于数据分析和操作
import pandas as pd

# 再次导入PySide6的部分模块，可能是由于代码重构或其他原因
from PySide6.QtGui import QGuiApplication
from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex

# 设置matplotlib的全局配置参数

plt.rcParams['font.family'] = 'serif'  # 设置全局字体为serif类型
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']  # 设置serif类型的字体列表，优先使用'Times New Roman'
plt.rcParams['svg.fonttype'] = 'none'  # 设置在保存为SVG格式的图像时，不将文本转换为路径
plt.rcParams['pdf.fonttype'] =  'truetype'  # 设置在保存为PDF格式的图像时，使用TrueType字体

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

# 获取当前文件的目录
current_directory = os.path.dirname(current_file_path)
working_directory = os.path.dirname(current_file_path)
# 改变当前工作目录
os.chdir(current_directory)

filename = 'Corrected/Remove_LOI_GeoRoc.db'
rock_type = 'VOL'
output_dir='TAS'


# 连接到数据库
conn = sqlite3.connect(filename)

# Read the data from the TAS_Data table
df = pd.read_sql_query("SELECT * FROM Current_Data", conn)

# Rename columns
df = df.rename(columns={
    "SIO2(WT%)": "SiO2(wt%)",
    "NA2O(WT%)": "Na2O(wt%)",
    "K2O(WT%)": "K2O(wt%)",
    "ROCK TYPE": "Rock Type"
})

selected_columns = df[["Type", "SiO2(wt%)", "Na2O(wt%)", "K2O(wt%)", "Rock Type"]]

# 筛选'Rock Type'为tag的行，并且去掉不含SiO2的行
tag_df = selected_columns[(selected_columns["Rock Type"] == rock_type) & (selected_columns["SiO2(wt%)"] != 0)]

# 删除含有NaN的行
tag_df = tag_df.dropna()

def sample_data(df, n_samples):
    return df.sample(n=min(len(df), n_samples), replace=True)


# 检查是否存在tag_color_dict.json文件
if os.path.exists('Color_Config/'+rock_type+'_color_dict.json'):
    # 如果存在，从文件中读取tag_color_dict
    with open('Color_Config/'+rock_type+'_color_dict.json', 'r') as f:
        tag_color_dict = json.load(f)
else:
    # 如果不存在，创建新的tag_color_dict并保存到文件中
    type_set = set(tag_df['Type'].unique())
    cmap = cm.get_cmap('rainbow', len(type_set))
    tag_color_dict = {type: cmap(i) for i, type in enumerate(type_set)}
    with open('Color_Config/'+rock_type+'_color_dict.json', 'w') as f:
            json.dump(tag_color_dict, f)    

sampled_df = tag_df.groupby('Type', group_keys=False).apply(sample_data, n_samples=1000, include_groups= True)

# 对每行数据，根据'Type'从tag_color_dict获得Color值，并将其转换为十六进制颜色代码
sampled_df['Color'] = sampled_df['Type'].map(tag_color_dict).apply(mcolors.rgb2hex)

# 如果有未能映射的'Type'，为其分配默认颜色
sampled_df['Color'] = sampled_df['Color'].fillna('b')  # 默认颜色为蓝色

print(sampled_df.head())

# 将'Type'列重命名为'Label'
sampled_df = sampled_df.rename(columns={'Type': 'Label'})

sampled_df.to_csv('sampled_TAS_data.csv', index=False)