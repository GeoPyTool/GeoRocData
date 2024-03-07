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


def Pearce_base(filename = 'GeoRoc.db',rock_type = 'VOL',output_dir='Pearce'):

    result_list = [['Label','Probability']]
    
    fig, ax_list = plt.subplots(2, 2, figsize=(10, 10))

    for ax in ax_list.flat:
        ax.set_aspect('equal', 'box')
    # 设置axes的宽高比为3:2
    # ax.set_aspect(2/3)

    # Record the start time
    begin_time = time.time()
    start_time = time.time()

    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)

    # 获取当前文件的目录
    current_directory = os.path.dirname(current_file_path)
    # 改变当前工作目录
    os.chdir(current_directory)


    with open(current_directory+'/Plot_Json/pearce_cord.json', 'r', encoding='utf-8') as file:
        cord = json.load(file)


    # 连接到数据库
    conn = sqlite3.connect(filename)

    # Read the data from the Pearce_Data table
    df = pd.read_sql_query("SELECT * FROM Current_Data", conn)

    Elements_List =["Y(PPM)", "NB(PPM)", "RB(PPM)", "YB(PPM)", "TA(PPM)"]
    # 筛选'ROCK TYPE'为tag的行，并且去掉数据不完整的行
    selected_columns = df[["Type", "Y(PPM)", "NB(PPM)", "RB(PPM)", "YB(PPM)", "TA(PPM)","ROCK TYPE"]]
    conditions = (selected_columns["ROCK TYPE"] == rock_type)
    for element in Elements_List:
        conditions = conditions & (selected_columns[element] != 0)

    tag_df = selected_columns[conditions]

    def title_except_in_parentheses(s):
        parts = s.split('(')
        return parts[0].title() + '(' + parts[1].upper() if len(parts) > 1 else parts[0].title()

    tag_df = tag_df.rename(columns=title_except_in_parentheses)

    tag_df["Y+Nb(PPM)"] = tag_df["Y(PPM)"] + tag_df["Nb(PPM)"]
    tag_df["Yb+Ta(PPM)"] = tag_df["Yb(PPM)"] + tag_df["Ta(PPM)"]

    target_x_list = ['Y+Nb(PPM)','Yb+Ta(PPM)','Y(PPM)','Yb(PPM)']
    target_y_list = ['Rb(PPM)','Rb(PPM)','Nb(PPM)','Ta(PPM)']   

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
    
    # 检查是否存在'Pearce_Base_' + tag + '_Withlines.pkl'文件
    if os.path.exists(output_dir+'/'+'Pearce_Base_' + rock_type + '_Withlines.pkl'):
        # 如果存在，从文件中读取tag_color_dict
        with open(output_dir+'/'+'Pearce_Base_' + rock_type + '_Withlines.pkl', 'rb') as f:
            fig = pickle.load(f)
    else:
        pass

        # 输出'Type'的取值个数
        # print(tag_df['Type'].value_counts())
        # 计算'Type'的取值个数
       
        # 绘制Pearce图解散点图
        # label = tag_df['Type']
        # 假设df是包含'x', 'y', 'label'列的DataFrame
        labelled_groups = set()
        grouped = tag_df.groupby('Type')

        label_locations = {}
        highest_y = 0

        # 遍历ax_list和target_y_list
        for ax, target_x,target_y in itertools.zip_longest(ax_list.flatten(), target_x_list, target_y_list):
            # 检查target_y的值，如果是None，就跳过绘图
            if target_y is None:
                ax.clear()
                ax.axis('off')
                continue

            # # 在每一个subplot上绘图
            # ax.plot(df['SiO2(wt%)'], df[target_y])
            labelled_groups = set()
            grouped = tag_df.groupby('Type')

            label_locations = {}
            highest_y = 0
            for label, group in grouped:
                x = tag_df[target_x]
                y = tag_df[target_y] 
                center_x = x.mean()
                center_y = y.mean()
                
                if label not in labelled_groups:            
                    labelled_groups.add(label)                    
                
                    data_amount = len(x)
                    # print(label, data_amount)
                    if(data_amount>30):

                        original_color =  mcolors.to_rgba(tag_color_dict[label])

                        # 定义一个基数，这个基数可以根据具体需求来调整
                        base = 0.08
                        # 计算透明度
                        alpha = base / np.log10(data_amount/10)             
                        
                        label_locations[label] = [center_x,center_y,original_color,alpha]
                        ax.scatter(x, y, color = original_color, edgecolors='none',  alpha = alpha)

                        # Record the end time
                        tmp_time = time.time()

                        # Calculate the time taken
                        time_taken = tmp_time - start_time
                        start_time = tmp_time
                        
                        print(f"{label} Data amount is {data_amount}, Alpha is {alpha:.3f}, Time taken: {time_taken:.3f} seconds")
                    
                            

            try:
                # 获取当前ax对象中的所有数据点
                for child in ax.get_children():
                    # 检查这个子对象是否是一个散点图的集合
                    if isinstance(child, collections.PathCollection):
                        # 获取当前透明度
                        current_alpha = child.get_alpha()
                        # 获取数据点的数量
                        num_points = child.get_sizes().size
                        # 根据当前透明度和数据点的数量设置新的透明度
                        if current_alpha is not None:
                            if num_points <1000:  # 如果数据点的数量大于100
                                child.set_alpha(min(current_alpha * 2, 0.3))  # 提高透明度，但不超过1
                            elif num_points >3000:  # 如果数据点的数量小于50
                                child.set_alpha(max(current_alpha / 2, 0.005))  # 降低透明度，但不低于0.01

                ax.set_xlabel(target_x)
                ax.set_ylabel(target_y)
                # 旋转y轴的标签
                ax.yaxis.set_tick_params(rotation=90)
            except KeyError:
                pass
        
        # 根据 center_x 对 label_locations 进行排序
        label_locations = dict(sorted(label_locations.items(), key=lambda item: item[1][0]))

        # 定义一个列表，包含n个值
        # 定义一个列表，步长为0.8
        values = list(np.arange(0, 7, 0.8))
        ha_list = ["left","right", "center", "left","right", "center"]
        va_list = ["bottom", "center", "top"]
        for i, (label, (center_x, center_y, original_color,alpha)) in enumerate(label_locations.items()):
            # 根据索引在 values 中轮流取值
            value = values[i % len(values)]
            ha = ha_list[i % len(ha_list)]
            va = va_list[i % len(va_list)]
            if(label =='Phonotephrite'):
                ha = 'left'
            # 在图中绘制文本
            used_x = center_x
            used_y = 18+value
            if highest_y <= used_y:
                highest_y = used_y
            ax.text(used_x, used_y, label, fontsize=11.5, color='k', 
                bbox=dict(facecolor=original_color, edgecolor=None, alpha= 0.3, pad=2),rotation = 0,
                horizontalalignment=ha, verticalalignment=va,
                            rotation_mode="anchor")

            # 添加一根从文本到中心点的虚线
            arrowstyle = '-|>'  # 箭头样式
            connectionstyle = ConnectionStyle("Arc3,rad=-0.3")  # 连接样式
            ax.annotate('', xy=(center_x, center_y), xytext=(used_x, used_y),
                        arrowprops=dict(arrowstyle=arrowstyle, connectionstyle=connectionstyle, linestyle='dashed', color=original_color, alpha=0.3))

        # # 绘制Pearce图解边界线条
        # # Draw Pearce diagram boundary lines
        # for line in cord['coords'].values():
        #     x_coords = [point[0] for point in line]
        #     y_coords = [point[1] for point in line]
        #     ax.plot(x_coords, y_coords, color='black', linewidth=0.3)
            

        # # 在Pearce图解中添加岩石种类标签
        # # Add rock type labels in Pearce diagram
        # for label, coords in cord['coords'].items():
        #     x_coords = [point[0] for point in coords]
        #     y_coords = [point[1] for point in coords]
        #     x_center = sum(x_coords) / len(x_coords)
        #     y_center = sum(y_coords) / len(y_coords)
        #     ax.text(x_center, y_center, label, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.3), fontsize=14)

        ax.set_xlabel(target_x, fontsize=14)
        ax.set_ylabel(target_y, fontsize=14)
        # ax.set_title("Extended Pearce Diagram", fontsize=14)
        # ax.set_xlim(35,80)
        # ax.set_ylim(0,17.647826086956513)  
        # # 在 y=17.647826086956513 的位置画一条横线
        # ax.axhline(17.647826086956513, linestyle='-', color='black', linewidth=0.3)
        # print(highest_y)
        # ax.set_ylim(0,highest_y+1)  
        # # 设置y轴的刻度
        # ax.set_yticks(range(0, 18))

        ax.tick_params(axis='both', labelsize=12)
        # legend = ax.legend(loc='upper left', fontsize=4, bbox_to_anchor=(1, 1))

        # # 获取当前的视域范围
        # xlim = ax.get_xlim()
        # ylim = ax.get_ylim()

        # 计算在视域范围内的数据点的数量
        # visible_points = tag_df[(tag_df["Y(PPM)"] >= xlim[0]) & 
        #                         (tag_df["Y(PPM)"] <= xlim[1]) & 
        #                         (tag_df['ALL_Alkaline'] >= ylim[0]) & 
        #                         (tag_df['ALL_Alkaline'] <= ylim[1])]

        # num_visible_points = len(visible_points)
        num_visible_points = len(tag_df)

        # 在图上显示可见的数据点的数量
        ax.text(0.05, 0.95, f'Used points: {num_visible_points}', transform=ax.transAxes, verticalalignment='top',fontsize=14)

        fig.tight_layout()

        with open(output_dir+'/'+'Pearce_Base_' + rock_type + '_Withlines.pkl', 'wb') as f:
            pickle.dump(fig, f)

    all_end_time = time.time()

    all_time_taken = all_end_time - begin_time


    # 保存图，包含图例
    # 创建存图的文件夹
    fig.savefig(output_dir+'/'+'Pearce_Base_' + rock_type + '.svg')
    # fig.savefig(output_dir+'/'+'Pearce_Base_' + rock_type + '.pdf')
    fig.savefig(output_dir+'/'+'Pearce_Base_' + rock_type + '.jpg', dpi=600)
    
    conn.close()
    print(f"All time taken: {all_time_taken:.3f} seconds")
    return(fig)
    

# 文件名
filename = 'GeoRoc.db'
rock_type = 'VOL'
color_config_dir = 'Color_Config'
if not os.path.exists(color_config_dir):
    os.makedirs(color_config_dir)
output_dir = 'Pearce'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

Pearce_base(filename, 'VOL', output_dir = 'Pearce')
Pearce_base(filename, 'PLU', output_dir = 'Pearce')