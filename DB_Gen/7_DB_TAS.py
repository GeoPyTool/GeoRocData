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


with open(current_directory+'/Plot_Json/tas_cord.json', 'r', encoding='utf-8') as file:
    cord = json.load(file)


def TAS_all(filename, rock_type, output_dir):
    # 连接到数据库
    conn = sqlite3.connect(filename)
    num_visible_points = 0

    # 创建一个宽高比为2:1的figure
    fig = plt.figure(figsize=(10, 5))     
    ax = fig.add_subplot(1, 1, 1)
    # 设置axes的宽高比为3:2
    ax.set_aspect(3/2)


    # Read the data from the TAS_Data table
    df = pd.read_sql_query("SELECT * FROM Current_Data", conn)

    selected_columns = df[["Type", "SIO2(WT%)", "NA2O(WT%)", "K2O(WT%)","ROCK TYPE"]]

    # 筛选'ROCK TYPE'为tag的行，并且去掉不含SiO2的行
    tag_df = selected_columns[(selected_columns["ROCK TYPE"] == rock_type) & (selected_columns["SIO2(WT%)"] != 0)]
    tag_df['ALL_Alkaline']= tag_df["NA2O(WT%)"]+tag_df["K2O(WT%)"]

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


    # 输出'Type'的取值个数
    print(tag_df['Type'].value_counts())
    # 计算'Type'的取值个数
    type_counts = tag_df['Type'].value_counts()
    type_counts.to_csv(output_dir+'/TAS_'+rock_type+'_type_counts_tag.csv')

    # 绘制TAS图解散点图
    # label = tag_df['Type']
    # 假设df是包含'x', 'y', 'label'列的DataFrame
    labelled_groups = set()
    grouped = tag_df.groupby('Type')

    for label, group in grouped:
        ax.scatter(group["SIO2(WT%)"], group['ALL_Alkaline'], alpha=0.05, label=label, color=tag_color_dict[label], edgecolors='none')
        center_x = group["SIO2(WT%)"].mean()
        center_y = group['ALL_Alkaline'].mean()
        if label not in labelled_groups:
            if 35 <= center_x <= 80 and 0 <= center_y <= 17.6478:
                # ax.text(center_x, center_y, label, fontsize=9)
                pass
            else:
                print(label+" Coordinates are out of bounds")
                labelled_groups.add(label)

    # 绘制TAS图解边界线条
    # Draw TAS diagram boundary lines
    for line in cord['coords'].values():
        x_coords = [point[0] for point in line]
        y_coords = [point[1] for point in line]
        ax.plot(x_coords, y_coords, color='black', linewidth=0.3)
        

    # 在TAS图解中添加岩石种类标签
    # Add rock type labels in TAS diagram
    for label, coords in cord['coords'].items():
        x_coords = [point[0] for point in coords]
        y_coords = [point[1] for point in coords]
        x_center = sum(x_coords) / len(x_coords)
        y_center = sum(y_coords) / len(y_coords)
        ax.text(x_center, y_center, label, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.3), fontsize=12)

    ax.set_xlabel(r"SiO2", fontsize=16)
    ax.set_ylabel(r"Na2O+K2O", fontsize=16)
    ax.set_title(r"Extended TAS Diagram", fontsize=16)
    ax.set_xlim(35,80)
    ax.set_ylim(0,17.647826086956513)  

    ax.tick_params(axis='both', labelsize=12)
    # legend = ax.legend(loc='upper left', fontsize=4, bbox_to_anchor=(1, 1))
    legend = ax.legend(loc='upper left', fontsize=4, bbox_to_anchor=(1, 1), ncol=2)

    # 获取当前的视域范围
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 计算在视域范围内的数据点的数量
    visible_points = tag_df[(tag_df["SIO2(WT%)"] >= xlim[0]) & (tag_df["SIO2(WT%)"] <= xlim[1]) & 
                            (tag_df['ALL_Alkaline'] >= ylim[0]) & (tag_df['ALL_Alkaline'] <= ylim[1])]

    num_visible_points = len(visible_points)

    # 在图上显示可见的数据点的数量
    ax.text(0.05, 0.95, f'Visible points: {num_visible_points}', transform=ax.transAxes, verticalalignment='top')

    fig.tight_layout()
    # 保存图，包含图例
    fig.savefig(output_dir+'/TAS_figure'+rock_type+'.svg')
    fig.savefig(output_dir+'/TAS_figure'+rock_type+'.png', dpi=600)
    # plt.show()
    
    conn.close()

# 文件名
filename = 'GeoRoc.db'
rock_type = 'VOL'
color_config_dir = 'Color_Config'
if not os.path.exists(color_config_dir):
    os.makedirs(color_config_dir)
output_dir = 'TAS'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

TAS_all(filename, 'VOL', output_dir = 'TAS')
TAS_all(filename, 'PLU', output_dir = 'TAS')