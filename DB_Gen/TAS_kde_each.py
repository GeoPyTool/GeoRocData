import configparser
import csv
import json
import math
import os
import pickle
import platform
import sqlite3
import time

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] =  'truetype'

# tag = 'VOL'
# # 文件名

# filename = 'GeoRoc.db'

def TAS_each(filename = 'Corrected/Remove_LOI_GeoRoc.db', rock_type = 'VOL',output_dir='TAS'):

    result_list = [['Label','Probability']]

    # Record the start time
    start_time = time.time()

    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)

    # 获取当前文件的目录
    current_directory = os.path.dirname(current_file_path)
    # 改变当前工作目录
    os.chdir(current_directory)

   
    with open(current_directory+'/Plot_Json/tas_cord.json', 'r', encoding='utf-8') as file:
        cord = json.load(file)


    # 连接到数据库
    conn = sqlite3.connect(filename)

    # Read the data from the TAS_Data table
    df = pd.read_sql_query("SELECT * FROM Current_Data", conn)

    selected_columns = df[["Type", "SIO2(WT%)", "NA2O(WT%)", "K2O(WT%)","ROCK TYPE"]]

    # 筛选'ROCK TYPE'为tag的行，并且去掉不含SiO2的行
    tag_df = selected_columns[(selected_columns["ROCK TYPE"] == rock_type) & (selected_columns["SIO2(WT%)"] != 0)]
    tag_df["ALL_Alkaline(WT%)"]= tag_df["NA2O(WT%)"]+tag_df["K2O(WT%)"]

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
    # print(tag_df['Type'].value_counts())
    # 计算'Type'的取值个数
    type_counts = tag_df['Type'].value_counts()
    type_counts.to_csv(output_dir+'/TAS_type_counts_'+ rock_type +'_.csv')

    # 绘制TAS图解散点图
    # label = tag_df['Type']
    # 假设df是包含'x', 'y', 'label'列的DataFrame
    labelled_groups = set()
    grouped = tag_df.groupby('Type')


    for label, group in grouped:
        # if label == 'Granite':
        #     pass
        # 创建一个宽高比为2:1的figure
        fig = plt.figure(figsize=(10, 5))     
        ax = fig.add_subplot(1, 1, 1)
        # 设置axes的宽高比为3:2
        ax.set_aspect(3/2)
        
        x = group[ "SIO2(WT%)"]
        y = group["ALL_Alkaline(WT%)"]
        # 删除包含NaN的行
        df = pd.DataFrame({'x': x, 'y': y})
        df = df.dropna()

        x = df['x']
        y = df['y']
        center_x = x.mean()
        center_y = y.mean()
        
        if label not in labelled_groups:
            if 35 <= center_x <= 80 and 0 <= center_y <= 17.6478:
                ax.text(center_x, center_y, label, fontsize=9)
                data_amount = len(x)
                print(label, data_amount)
                if(data_amount>=964):          
                    # Construct the file path
                    data = np.column_stack((x, y))
                    file_path = rock_type + '_GMM_kde/'+label+'_kde.pkl'
                    # Check if the file exists
                    if os.path.exists(file_path):
                        pass
                        # # If the file exists, open it and load the data file in binary mode and load the object
                        with open(file_path , 'rb') as f:
                            kde = pickle.load(f)
                    else:
                        pass
                        # 使用核密度估计对数据进行拟合
                        kde = KernelDensity(kernel='gaussian').fit(data)

                        # 计算数据点的概率密度
                        # 将概率场KDE保存为文件
                        os.makedirs(rock_type + '_GMM_kde', exist_ok=True)                            
                        # Save the KDE object to a file
                        with open(rock_type + '_GMM_kde/'+label+'_kde.pkl', 'wb') as f:
                            pickle.dump(kde, f)
        
                    
                    probs = np.exp(kde.score_samples(data))

                    # Convert densities to probabilities
                    max_probs = max(probs)

                    # Normalize probabilities
                    probs/= max_probs 

                    # 获取原始颜色
                    original_color =  mcolors.to_rgba(tag_color_dict[label])

                    # 将原始颜色与白色混合，使颜色更浅
                    lighter_color = (0.6 * np.array(mcolors.to_rgba('white')) + 0.5 * np.array(original_color)).tolist()
                    # 将原始颜色与灰色混合，使颜色更深
                    darker_color = (0.6 * np.array(mcolors.to_rgba('gray')) + 0.5 * np.array(original_color)).tolist()                        
                    cmap = mcolors.LinearSegmentedColormap.from_list('custom', [lighter_color,darker_color], N=64)
                    
                    # ax.scatter(x, y, c=probs, label=label, cmap= cmap,  edgecolors='none')

                    # 创建一个网格
                    n = 1024
                    # x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), n), np.linspace(min(y), max(y), n))
                    x_grid, y_grid = np.meshgrid(np.linspace(35, 90, n), np.linspace(0,20, n))

                    # 将网格转换为二维数组
                    grid_points = np.array([x_grid.ravel(), y_grid.ravel()]).T

                    # 计算网格上每个点的对数概率密度
                    log_prob_grid = kde.score_samples(grid_points)

                    # 将对数概率密度转换为概率密度, 将每个概率密度除以峰值，进行归一化
                    prob_grid = np.exp(log_prob_grid)/max_probs

                    # 将概率密度重新塑形为网格的形状
                    prob_grid = prob_grid.reshape(x_grid.shape)

                    # 绘制等高线
                    contour = ax.contour(x_grid, y_grid, prob_grid, cmap=cmap)

                    # 在等高线上添加概率密度的值
                    ax.clabel(contour, inline=True, fontsize=7)

                    # 定义一个基数，这个基数可以根据具体需求来调整
                    base = 0.08
                    # 计算透明度
                    alpha = base / np.log10(data_amount/10)                  
                    
                    # ax.scatter(x, y, c=probs, label=label, cmap='terrain', edgecolors='none')
                    ax.scatter(x, y, color = original_color, edgecolors='none',  alpha = alpha)

                    # Record the end time
                    end_time = time.time()

                    # Calculate the time taken
                    time_taken = end_time - start_time
                    start_time = end_time
                    print(f"{label} Data amount is {data_amount}, Alpha is {alpha:.3f}, Time taken: {time_taken:.3f} seconds")
                   
                    # 计算新点的类别概率
                    new_point = np.array([[50,8]])  # 新点
                    # new_point = new_point.reshape(1, -1)  # 将new_point重新塑形为二维数组
                    # 计算新点的对数概率密度
                    log_prob = kde.score_samples(new_point)
                    # 将对数概率密度转换为概率密度
                    new_point_prob = np.exp(log_prob)
                    # print(f"The probability of {label} is {new_point_prob[0]:.2f}")
                    # print(f"The probability of {label} is {new_point_prob.tolist()}")

                    result_list.append([label,new_point_prob[0]])

                    # 绘制TAS图解边界线条
                    # Draw TAS diagram boundary lines
                    for line in cord['coords'].values():
                        x_coords = [point[0] for point in line]
                        y_coords = [point[1] for point in line]
                        ax.plot(x_coords, y_coords, color='black', linewidth=0.3)
                        

                    # 在TAS图解中添加岩石种类标签
                    # Add rock type labels in TAS diagram
                    # for label, coords in cord['coords'].items():
                    #     x_coords = [point[0] for point in coords]
                    #     y_coords = [point[1] for point in coords]
                    #     x_center = sum(x_coords) / len(x_coords)
                    #     y_center = sum(y_coords) / len(y_coords)
                    #     ax.text(x_center, y_center, label, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.3), fontsize=9)

                    ax.set_xlabel(r"$SiO2$", fontsize=9)
                    ax.set_ylabel(r"$Na2O+K2O$", fontsize=9)
                    ax.set_title(r"TAS Diagram", fontsize=9)
                    ax.set_xlim(35,80)
                    ax.set_ylim(0,17.647826086956513)  

                    ax.tick_params(axis='both', labelsize=9)
                    # legend = ax.legend(loc='upper left', fontsize=4, bbox_to_anchor=(1, 1))

                    # 获取当前的视域范围
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()

                    # 计算在视域范围内的数据点的数量
                    visible_points = tag_df[(tag_df["SIO2(WT%)"] >= xlim[0]) & 
                                            (tag_df["SIO2(WT%)"] <= xlim[1]) & 
                                            (tag_df["ALL_Alkaline(WT%)"] >= ylim[0]) & 
                                            (tag_df["ALL_Alkaline(WT%)"] <= ylim[1])]

                    num_visible_points = len(visible_points)

                    # 在图上显示可见的数据点的数量
                    ax.text(0.8, 0.98, f'Used points: {num_visible_points}', transform=ax.transAxes, verticalalignment='top')
                    # ax.text(0.75, 0.98, f'Used points: {num_visible_points}', transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=14)
        
                    fig.tight_layout()
                    # 保存图，包含图例
                    # 创建存图的文件夹
                    os.makedirs(rock_type + '_GMM_svg', exist_ok=True)
                    os.makedirs(rock_type + '_GMM_pdf', exist_ok=True)
                    os.makedirs(rock_type + '_GMM_jpg', exist_ok=True)
                    fig.savefig(rock_type +'_GMM_svg/'+ label +'.svg')
                    fig.savefig(rock_type +'_GMM_pdf/'+ label +'.pdf')
                    fig.savefig(rock_type +'_GMM_jpg/'+ label +'.jpg', dpi=600)
                    # plt.show()

            else:
                # print(label+" Coordinates are out of bounds")
                pass
            labelled_groups.add(label)
    pd.DataFrame(result_list).to_csv('TAS_test_prob_'+ rock_type +'.csv',index=False)
    conn.close()

TAS_each('Corrected/Remove_LOI_GeoRoc.db','VOL')
TAS_each('Corrected/Remove_LOI_GeoRoc.db','PLU')