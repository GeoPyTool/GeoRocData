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

filename = 'TAS Result_1000Samples.csv'

df = pd.read_csv(filename,encoding='utf-8')

# print(df.columns)

# 比较'Label'列和'Max_Ratio Classification'列的值，将结果保存为'Max_Ratio Test'列
df['Max_Ratio Test'] = df.apply(lambda row: row['Label'] == row['Max_Ratio Classification'], axis=1)

# 比较'Label'列和'Soft-Max Classification'列的值，将结果保存为'Soft-Max Test'列
df['Soft-Max Test'] = df.apply(lambda row: row['Label'] == row['Soft-Max Classification'], axis=1)

# 比较'Label'列和'TAS as VOL'列的值，将结果保存为'TAS Test'列
df['TAS Test'] = df.apply(lambda row: row['Label'] == row['TAS as VOL'], axis=1)


# 定义新的列顺序
new_order = ['Label', 'Max_Ratio Classification', 'Max_Ratio Probability','Max_Ratio Test', 
             'Soft-Max Classification', 'Soft-Max Probability', 'Soft-Max Test', 'TAS as VOL','TAS Test',
             'SiO2(wt%)', 'Na2O(wt%)', 'K2O(wt%)', 'Rock Type', 'Color', 'Marker']

# 重新排列列
df = df.reindex(columns=new_order)

df.to_csv('sampled_TAS_data_evaluated.csv', index=False)

# 对每个测试列进行计算
for test in ['Max_Ratio Test', 'Soft-Max Test', 'TAS Test']:
    true_ratio = df[test].mean()
    print(f"{test} True ratio: {true_ratio}")

# 创建一个新的DataFrame来保存结果
true_ratio_df = pd.DataFrame()

# 对每个测试列进行计算
for test in ['Max_Ratio Test', 'Soft-Max Test', 'TAS Test']:
    true_ratio_df[test] = df.groupby('Label')[test].mean().round(3)

# 删除所有值都为0的行
true_ratio_df = true_ratio_df.loc[~(true_ratio_df==0).all(axis=1)]

# 查看结果
print(true_ratio_df)

true_ratio_df.to_csv('sampled_TAS_data_evaluated_result.csv', index= True)