import configparser
import csv
import json
import math
import os
import platform
import sqlite3
import pickle
import time

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.patches import ConnectionStyle
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import scipy as sp
import toga
import toga_chart

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] =  'truetype'
from toga.style.pack import COLUMN, ROW
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import json

# tag = 'VOL'
# # 文件名

# filename = 'GeoRoc_Database.db'

def TAS_each(filename = 'GeoRoc_Database.db',tag = 'VOL'):

    result_list = [['Label','Probability']]

    # Record the start time
    begin_time = time.time()
    start_time = time.time()

    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)

    # 获取当前文件的目录
    current_directory = os.path.dirname(current_file_path)
    # 改变当前工作目录
    os.chdir(current_directory)

    with open('tas_cord.json') as file:
        cord = json.load(file)

    # 连接到数据库
    conn = sqlite3.connect(filename)

    # Read the data from the TAS_Data table
    df = pd.read_sql_query("SELECT * FROM Major_Data", conn)

    # 选择'SIO2_wt_calibred'，'ALL_Alkaline_wt_calibred'，'Type'，'Date'，'Latitude'，'Longitude'这些列
    selected_columns = df[['Type', 'Date', 'Latitude', 'Longitude','SIO2_wt_calibred', 'ALL_Alkaline_wt_calibred','ROCK_TYPE']]

    # 筛选'ROCK_TYPE'为tag的行，并且去掉不含SiO2的行
    tag_df = selected_columns[(selected_columns['ROCK_TYPE'] == tag) & (selected_columns['SIO2_wt_calibred'] != 0)]

    # 检查是否存在tag_color_dict.json文件
    if os.path.exists(tag+'_color_dict.json'):
        # 如果存在，从文件中读取tag_color_dict
        with open(tag+'_color_dict.json', 'r') as f:
            tag_color_dict = json.load(f)
    else:
        # 如果不存在，创建新的tag_color_dict并保存到文件中
        type_set = set(tag_df['Type'].unique())
        cmap = cm.get_cmap('rainbow', len(type_set))
        tag_color_dict = {type: cmap(i) for i, type in enumerate(type_set)}
        with open(tag+'_color_dict.json', 'w') as f:
            json.dump(tag_color_dict, f)

    # 输出'Type'的取值个数
    # print(tag_df['Type'].value_counts())
    # 计算'Type'的取值个数
    type_counts = tag_df['Type'].value_counts()
    type_counts.to_csv('TAS_type_counts_'+tag+'_.csv')

    # 绘制TAS图解散点图
    # label = tag_df['Type']
    # 假设df是包含'x', 'y', 'label'列的DataFrame
    labelled_groups = set()
    grouped = tag_df.groupby('Type')
    # 创建一个指定宽高比的figure
    # fig = plt.figure(figsize=(10, 10))  
    fig = plt.figure(figsize=(10, 10))     
    ax = fig.add_subplot(1, 1, 1)
    # 设置axes的宽高比为3:2
    # ax.set_aspect(2/3)

    label_locations = {}
    highest_y = 0

    for label, group in grouped:
        x = group['SIO2_wt_calibred']
        y = group['ALL_Alkaline_wt_calibred']
        center_x = x.mean()
        center_y = y.mean()
        
        if label not in labelled_groups:            
            labelled_groups.add(label)
            
            if 35 <= center_x <= 80 and 0 <= center_y <= 17.6478:
                
                data_amount = len(x)
                # print(label, data_amount)
                if(data_amount>30):

                    original_color =  mcolors.to_rgba(tag_color_dict[label])
                    
                    label_locations[label] = [center_x,center_y,original_color]

                    
                    # # ax.text(center_x, center_y, label, fontsize=7)
                    # # ax.text(center_x, center_y, label, fontsize=7, rotation=45)
                    # ax.text(center_x, center_y, label, fontsize=7, color=original_color, 
                    #     bbox=dict(facecolor='white', alpha=0.5))
                    

                    try:           
                        
                        data = np.column_stack((x, y))    
                        # Construct the file path
                        # file_path = tag + '_GMM_kde/'+label+'_kde.pkl'
                        # # Check if the file exists
                        # if os.path.exists(file_path):
                        #     # If the file exists, open it and load the data file in binary mode and load the object
                        #     with open(file_path, 'rb') as f:
                        #         kde = pickle.load(f)
                        # else:
                        #     data = np.column_stack((x, y))

                        #     # 使用核密度估计对数据进行拟合
                        #     kde = KernelDensity(kernel='gaussian').fit(data)

                        #     # 将概率场KDE保存为文件
                        #     os.makedirs(tag + '_GMM_half_kde', exist_ok=True)                            
                        #     # Save the KDE object to a file
                        #     with open(tag + '_GMM_half_kde/'+label+'_kde.pkl', 'wb') as f:
                        #         pickle.dump(kde, f)
            
                        # # 计算数据点的概率密度
                        # probs = np.exp(kde.score_samples(data))
                        # 获取原始颜色

                        # 将原始颜色与白色混合，使颜色更浅
                        lighter_color = (0.5 * np.array(mcolors.to_rgba('white')) + 0.5 * np.array(original_color)).tolist()
                        # 将原始颜色与灰色混合，使颜色更深
                        darker_color = (0.5 * np.array(mcolors.to_rgba('gray')) + 0.5 * np.array(original_color)).tolist()                        
                        cmap = mcolors.LinearSegmentedColormap.from_list('custom', [darker_color, lighter_color], N=64)
                        
                        # 定义一个基数，这个基数可以根据具体需求来调整
                        base = 0.08
                        # 计算透明度
                        alpha = base / np.log10(data_amount/10)
                    
                        ax.scatter(x, y, color = original_color, edgecolors='none',  alpha = alpha)

                        # Record the end time
                        tmp_time = time.time()

                        # Calculate the time taken
                        time_taken = tmp_time - start_time
                        start_time = tmp_time

                        
                        print(f"{label} Data amount is {data_amount}, Alpha is {alpha:.3f}, Time taken: {time_taken:.3f} seconds")
                   
                        # # 计算新点的类别概率
                        # new_point = np.array([[50,8]])  # 新点
                        # # new_point = new_point.reshape(1, -1)  # 将new_point重新塑形为二维数组
                        # # 计算新点的对数概率密度
                        # log_prob = kde.score_samples(new_point)
                        # # 将对数概率密度转换为概率密度
                        # new_point_prob = np.exp(log_prob)
                        # # print(f"The probability of {label} is {new_point_prob[0]:.2f}")
                        # # print(f"The probability of {label} is {new_point_prob.tolist()}")
                        

                        # result_list.append([label,new_point_prob[0]])
                       
                    except ValueError:
                        print(label," Insufficient for Gaussian KDE.")
                        kde = None
            else:
                # print(label+" Coordinates are out of bounds")
                pass

    
    # 根据 center_x 对 label_locations 进行排序
    label_locations = dict(sorted(label_locations.items(), key=lambda item: item[1][0]))

    # 定义一个列表，包含n个值
    # 定义一个列表，步长为0.8
    values = list(np.arange(0, 7, 0.8))
    ha_list = ["left","right", "center", "left","right", "center"]
    va_list = ["bottom", "center", "top"]
    for i, (label, (center_x, center_y, original_color)) in enumerate(label_locations.items()):
        # 根据索引在 values 中轮流取值
        value = values[i % len(values)]
        ha = ha_list[i % len(ha_list)]
        va = va_list[i % len(va_list)]
        # 在图中绘制文本
        used_x = center_x
        used_y = 18+value
        if highest_y <= used_y:
            highest_y = used_y
        # ax.text(used_x, used_y, label, fontsize=6.5, color='k', 
        #     bbox=dict(facecolor=original_color, edgecolor=None, alpha=0.6, pad=2),rotation = 45)
        ax.text(used_x, used_y, label, fontsize=12, color='k', 
            bbox=dict(facecolor=original_color, edgecolor=None, alpha=0.6, pad=2),rotation = 0,
              horizontalalignment=ha, verticalalignment=va,
                         rotation_mode="anchor")

        # 添加一根从文本到中心点的虚线
        arrowstyle = '-|>'  # 箭头样式
        connectionstyle = ConnectionStyle("Arc3,rad=-0.3")  # 连接样式
        ax.annotate('', xy=(center_x, center_y), xytext=(used_x, used_y),
                    arrowprops=dict(arrowstyle=arrowstyle, connectionstyle=connectionstyle, linestyle='dashed', color=original_color, alpha=0.3))

        # ax.annotate('', xy=(center_x, center_y), xytext=(used_x, used_y),
        #             arrowprops=dict(arrowstyle='->', linestyle='dashed', alpha=0.3,color=original_color))
        
        # bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.5, pad=5)

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
        ax.text(x_center, y_center, label, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.3), fontsize=14)

    ax.set_xlabel(r"$SiO2$", fontsize=14)
    ax.set_ylabel(r"$Na2O+K2O$", fontsize=14)
    ax.set_title(r"Extended TAS Diagram", fontsize=14)
    ax.set_xlim(35,80)
    # ax.set_ylim(0,17.647826086956513)  
    # 在 y=17.647826086956513 的位置画一条横线
    ax.axhline(17.647826086956513, linestyle='-', color='black', linewidth=0.3)
    print(highest_y)
    ax.set_ylim(0,highest_y+1)  
    # 设置y轴的刻度
    ax.set_yticks(range(0, 18))

    ax.tick_params(axis='both', labelsize=9)
    # legend = ax.legend(loc='upper left', fontsize=4, bbox_to_anchor=(1, 1))

    # 获取当前的视域范围
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 计算在视域范围内的数据点的数量
    visible_points = tag_df[(tag_df['SIO2_wt_calibred'] >= xlim[0]) & 
                            (tag_df['SIO2_wt_calibred'] <= xlim[1]) & 
                            (tag_df['ALL_Alkaline_wt_calibred'] >= ylim[0]) & 
                            (tag_df['ALL_Alkaline_wt_calibred'] <= ylim[1])]

    num_visible_points = len(visible_points)

    # 在图上显示可见的数据点的数量
    ax.text(0.05, 0.95, f'Used points: {num_visible_points}', transform=ax.transAxes, verticalalignment='top')

    fig.tight_layout()
    # 保存图，包含图例
    # 创建存图的文件夹
    fig.savefig('TAS_figure_' + tag + '_nolines.svg')
    fig.savefig('TAS_figure_' + tag + '_nolines.png', dpi=600)
    all_end_time = time.time()

    all_time_taken = all_end_time - begin_time

    
    print(f"All time taken: {all_time_taken:.3f} seconds")

    
    conn.close()

TAS_each('GeoRoc_Database.db','VOL')
TAS_each('GeoRoc_Database.db','PLU')