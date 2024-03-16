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
from matplotlib.patches import ConnectionStyle
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


def TAS_base(filename = 'Corrected/Remove_LOI_GeoRoc.db',rock_type = 'VOL',output_dir='TAS'):

    result_list = [['Label','Probability']]
    
    # 创建一个指定宽高比的figure
    fig = plt.figure(figsize=(10, 10))     
    ax = fig.add_subplot(1, 1, 1)
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
    
    # 检查是否存在'TAS_Base_' + tag + '_Withlines.pkl'文件
    if os.path.exists(output_dir+'/'+'TAS_Base_' + rock_type + '_Withlines.pkl'):
        # 如果存在，从文件中读取tag_color_dict
        with open(output_dir+'/'+'TAS_Base_' + rock_type + '_Withlines.pkl', 'rb') as f:
            fig = pickle.load(f)
    else:
        pass

        # 输出'Type'的取值个数
        # print(tag_df['Type'].value_counts())
        # 计算'Type'的取值个数
        type_counts = tag_df['Type'].value_counts()
        type_counts.to_csv(output_dir+'/'+'TAS_type_counts_'+ rock_type +'.csv')

        # 绘制TAS图解散点图
        # label = tag_df['Type']
        # 假设df是包含'x', 'y', 'label'列的DataFrame
        labelled_groups = set()
        grouped = tag_df.groupby('Type')

        label_locations = {}
        highest_y = 0

        for label, group in grouped:
            x = group["SIO2(WT%)"]
            y = group["ALL_Alkaline(WT%)"]
            center_x = x.mean()
            center_y = y.mean()
            
            if label not in labelled_groups:            
                labelled_groups.add(label)
                
                if 35 <= center_x <= 80 and 0 <= center_y <= 17.6478:
                    
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

        ax.set_xlabel("SiO2", fontsize=14)
        ax.set_ylabel("Na2O+K2O", fontsize=14)
        ax.set_title("TAS-PFS Diagram", fontsize=14)
        ax.set_xlim(35,80)
        # ax.set_ylim(0,17.647826086956513)  
        # 在 y=17.647826086956513 的位置画一条横线
        ax.axhline(17.647826086956513, linestyle='-', color='black', linewidth=0.3)
        print(highest_y)
        ax.set_ylim(0,highest_y+1)  
        # 设置y轴的刻度
        ax.set_yticks(range(0, 18))

        ax.tick_params(axis='both', labelsize=12)
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
        ax.text(0.75, 0.98, f'Used points: {num_visible_points}', transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=14)
        
        fig.tight_layout()

        with open(output_dir+'/'+'TAS_Base_' + rock_type + '_Withlines.pkl', 'wb') as f:
            pickle.dump(fig, f)

    all_end_time = time.time()

    all_time_taken = all_end_time - begin_time


    # 保存图，包含图例
    # 创建存图的文件夹
    fig.savefig(output_dir+'/'+'TAS_Base_' + rock_type + '.svg')
    # fig.savefig(output_dir+'/'+'TAS_Base_' + rock_type + '.pdf')
    fig.savefig(output_dir+'/'+'TAS_Base_' + rock_type + '.jpg', dpi=600)
    
    conn.close()
    print(f"All time taken: {all_time_taken:.3f} seconds")
    return(fig)
    

def TAS_No_Lines(filename = 'Corrected/Remove_LOI_GeoRoc.db',rock_type = 'VOL',output_dir='TAS'):

    result_list = [['Label','Probability']]
    
    # 创建一个指定宽高比的figure
    fig = plt.figure(figsize=(10, 10))     
    ax = fig.add_subplot(1, 1, 1)
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
    
    # 检查是否存在'TAS_Base_' + tag + '_Withlines.pkl'文件
    if os.path.exists(output_dir+'/'+'TAS_Base_' + rock_type + '_Nolines.pkl'):
        # 如果存在，从文件中读取tag_color_dict
        with open(output_dir+'/'+'TAS_Base_' + rock_type + '_Nolines.pkl', 'rb') as f:
            fig = pickle.load(f)
    else:
        pass

        # 输出'Type'的取值个数
        # print(tag_df['Type'].value_counts())
        # 计算'Type'的取值个数
        type_counts = tag_df['Type'].value_counts()
        type_counts.to_csv(output_dir+'/'+'TAS_type_counts_'+ rock_type +'.csv')

        # 绘制TAS图解散点图
        # label = tag_df['Type']
        # 假设df是包含'x', 'y', 'label'列的DataFrame
        labelled_groups = set()
        grouped = tag_df.groupby('Type')

        label_locations = {}
        highest_y = 0

        for label, group in grouped:
            x = group["SIO2(WT%)"]
            y = group["ALL_Alkaline(WT%)"]
            center_x = x.mean()
            center_y = y.mean()
            
            if label not in labelled_groups:            
                labelled_groups.add(label)
                
                if 35 <= center_x <= 80 and 0 <= center_y <= 17.6478:
                    
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
            # ax.text(used_x, used_y, label, fontsize=11.5, color='k', 
            #     bbox=dict(facecolor=original_color, edgecolor=None, alpha= 0.3, pad=2),rotation = 0,
            #     horizontalalignment=ha, verticalalignment=va,
            #                 rotation_mode="anchor")

            # # 添加一根从文本到中心点的虚线
            # arrowstyle = '-|>'  # 箭头样式
            # connectionstyle = ConnectionStyle("Arc3,rad=-0.3")  # 连接样式
            # ax.annotate('', xy=(center_x, center_y), xytext=(used_x, used_y),
            #             arrowprops=dict(arrowstyle=arrowstyle, connectionstyle=connectionstyle, linestyle='dashed', color=original_color, alpha=0.3))

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

        ax.set_xlabel(r"$SiO_2$", fontsize=14)
        ax.set_ylabel(r"$Na_2O+K_2O$", fontsize=14)
        ax.set_title(r"TAS Diagram", fontsize=14)
        ax.set_xlim(35,80)
        ax.set_ylim(0,17.647826086956513)  
        ax.tick_params(axis='both', labelsize=9)
        # 在 y=17.647826086956513 的位置画一条横线
        ax.axhline(17.647826086956513, linestyle='-', color='black', linewidth=0.3)
        print(highest_y)
        ax.set_ylim(0,highest_y+1)  
        # 设置y轴的刻度
        ax.set_yticks(range(0, 18))

        ax.tick_params(axis='both', labelsize=12)
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
        ax.text(0.75, 0.98, f'Used points: {num_visible_points}', transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=14)
        
        fig.tight_layout()

        with open(output_dir+'/'+'TAS_Base_' + rock_type + '_Nolines.pkl', 'wb') as f:
            pickle.dump(fig, f)

    all_end_time = time.time()

    all_time_taken = all_end_time - begin_time


    # 保存图，包含图例
    # 创建存图的文件夹
    fig.savefig(output_dir+'/'+'TAS_Base_' + rock_type + '_Nolines.svg')
    # fig.savefig(output_dir+'/'+'TAS_Base_' + rock_type + '.pdf')
    fig.savefig(output_dir+'/'+'TAS_Base_' + rock_type + '_Nolines.jpg', dpi=600)
    
    conn.close()
    print(f"All time taken: {all_time_taken:.3f} seconds")
    return(fig)
    

def TAS_No_Colors(filename = 'Corrected/Remove_LOI_GeoRoc.db',rock_type = 'VOL',output_dir='TAS'):

    result_list = [['Label','Probability']]
    
    # 创建一个指定宽高比的figure
    fig = plt.figure(figsize=(10, 10))     
    ax = fig.add_subplot(1, 1, 1)
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
    
    # 检查是否存在'TAS_Base_' + tag + '_Withlines.pkl'文件
    if os.path.exists(output_dir+'/'+'TAS_Base_' + rock_type + '_Nolines_Nocolors.pkl'):
        # 如果存在，从文件中读取tag_color_dict
        with open(output_dir+'/'+'TAS_Base_' + rock_type + '_Nolines_Nocolors.pkl', 'rb') as f:
            fig = pickle.load(f)
    else:
        pass

        # 输出'Type'的取值个数
        # print(tag_df['Type'].value_counts())
        # 计算'Type'的取值个数
        type_counts = tag_df['Type'].value_counts()
        type_counts.to_csv(output_dir+'/'+'TAS_type_counts_'+ rock_type +'.csv')

        # 绘制TAS图解散点图
        # label = tag_df['Type']
        # 假设df是包含'x', 'y', 'label'列的DataFrame
        labelled_groups = set()
        grouped = tag_df.groupby('Type')

        label_locations = {}
        highest_y = 0

        for label, group in grouped:
            x = group["SIO2(WT%)"]
            y = group["ALL_Alkaline(WT%)"]
            center_x = x.mean()
            center_y = y.mean()
            
            if label not in labelled_groups:            
                labelled_groups.add(label)
                
                if 35 <= center_x <= 80 and 0 <= center_y <= 17.6478:
                    
                    data_amount = len(x)
                    # print(label, data_amount)
                    if(data_amount>30):

                        original_color =  mcolors.to_rgba(tag_color_dict[label])

                        # 定义一个基数，这个基数可以根据具体需求来调整
                        base = 0.08
                        # 计算透明度
                        alpha = base / np.log10(data_amount/10)             
                        
                        label_locations[label] = [center_x,center_y,original_color,alpha]
                        # ax.scatter(x, y, color = original_color, edgecolors='none',  alpha = alpha)

                        # Record the end time
                        tmp_time = time.time()

                        # Calculate the time taken
                        time_taken = tmp_time - start_time
                        start_time = tmp_time

                        
                        print(f"{label} Data amount is {data_amount}, Alpha is {alpha:.3f}, Time taken: {time_taken:.3f} seconds")
                    
                        
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
            # ax.text(used_x, used_y, label, fontsize=11.5, color='k', 
            #     bbox=dict(facecolor=original_color, edgecolor=None, alpha= 0.3, pad=2),rotation = 0,
            #     horizontalalignment=ha, verticalalignment=va,
            #                 rotation_mode="anchor")

            # # 添加一根从文本到中心点的虚线
            # arrowstyle = '-|>'  # 箭头样式
            # connectionstyle = ConnectionStyle("Arc3,rad=-0.3")  # 连接样式
            # ax.annotate('', xy=(center_x, center_y), xytext=(used_x, used_y),
            #             arrowprops=dict(arrowstyle=arrowstyle, connectionstyle=connectionstyle, linestyle='dashed', color=original_color, alpha=0.3))

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

        ax.set_xlabel(r"$SiO_2$", fontsize=14)
        ax.set_ylabel(r"$Na_2O+K_2O$", fontsize=14)
        ax.set_title(r"TAS Diagram", fontsize=14)
        ax.set_xlim(35,80)
        ax.set_ylim(0,17.647826086956513)  
        ax.tick_params(axis='both', labelsize=9)
        # 在 y=17.647826086956513 的位置画一条横线
        ax.axhline(17.647826086956513, linestyle='-', color='black', linewidth=0.3)
        print(highest_y)
        ax.set_ylim(0,highest_y+1)  
        # 设置y轴的刻度
        ax.set_yticks(range(0, 18))

        ax.tick_params(axis='both', labelsize=12)
        # legend = ax.legend(loc='upper left', fontsize=4, bbox_to_anchor=(1, 1))

        # 获取当前的视域范围
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # # 计算在视域范围内的数据点的数量
        # visible_points = tag_df[(tag_df["SIO2(WT%)"] >= xlim[0]) & 
        #                         (tag_df["SIO2(WT%)"] <= xlim[1]) & 
        #                         (tag_df["ALL_Alkaline(WT%)"] >= ylim[0]) & 
        #                         (tag_df["ALL_Alkaline(WT%)"] <= ylim[1])]

        # num_visible_points = len(visible_points)

        # # 在图上显示可见的数据点的数量
        # ax.text(0.75, 0.98, f'Used points: {num_visible_points}', transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', fontsize=14)
        
        fig.tight_layout()

        with open(output_dir+'/'+'TAS_Base_' + rock_type + '_Nolines.pkl', 'wb') as f:
            pickle.dump(fig, f)

    all_end_time = time.time()

    all_time_taken = all_end_time - begin_time


    # 保存图，包含图例
    # 创建存图的文件夹
    fig.savefig(output_dir+'/'+'TAS_Base_' + rock_type + '_Nolines.svg')
    # fig.savefig(output_dir+'/'+'TAS_Base_' + rock_type + '.pdf')
    fig.savefig(output_dir+'/'+'TAS_Base_' + rock_type + '_Nolines.jpg', dpi=600)
    
    conn.close()
    print(f"All time taken: {all_time_taken:.3f} seconds")
    return(fig)
    


# 文件名
filename = 'GeoRoc.db'
rock_type = 'VOL'
color_config_dir = 'Color_Config'
if not os.path.exists(color_config_dir):
    os.makedirs(color_config_dir)
output_dir = 'TAS'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

TAS_base(filename, 'VOL', output_dir = 'TAS')
TAS_base(filename, 'PLU', output_dir = 'TAS')
TAS_No_Lines(filename, 'VOL', output_dir = 'TAS')
TAS_No_Lines(filename, 'PLU', output_dir = 'TAS')
TAS_No_Colors(filename, 'VOL', output_dir = 'TAS')
TAS_No_Colors(filename, 'PLU', output_dir = 'TAS')