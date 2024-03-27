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



def generate_polygon():
    Polygon_dict = {}
    
    # 从'Plot_Json/tas_cord.json'文件中加载数据
    with open('Plot_Json/tas_cord.json', 'r', encoding='utf-8') as file:
        cord = json.load(file)
    # 将读取的边界线条数据存储到类实例变量tas_cord中
    tas_cord = cord

    # 绘制TAS图解的所有边界线条
    # Draw all boundary lines for the TAS diagram
    for type_label, line in cord['coords'].items():
        # 提取每条线的x坐标和y坐标
        x_coords = [point[0] for point in line]
        y_coords = [point[1] for point in line]

        # 创建一个闭合的多边形，不填充颜色，边框颜色为红色
        polygon = Polygon(list(zip(x_coords, y_coords)), closed=True, fill=None, edgecolor='r')

        # 将创建好的多边形对象存入字典中，键为type_label
        Polygon_dict[type_label]=polygon
    return( Polygon_dict,tas_cord )


def plot_data(file_name = "Geochemistry.csv"):
    # 设置标签为'VOL'
    tag = 'VOL'
    
    # 设置设置为'Withlines'
    setting = 'Withlines'
    
    # 设置颜色设置为空字符串
    color_setting = ''
    
    # 设置数据路径为空字符串
    data_path = ''
    dpi = 100

    if file_name:
        # 根据文件扩展名判断并读取数据
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_name)  # 从CSV文件中读取数据
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_name)  # 从Excel文件中读取数据


    # 'TAS_Base_VOL_Nolines.pkl'
    pkl_filename='TAS/TAS_Base_'+tag+'_'+setting+color_setting+'.pkl'

    # Load the Figure
    with open(pkl_filename, 'rb') as f:
        fig = pickle.load(f)
        # print('fig loaded')
    # Create a new FigureCanvas

    # Get the Axes
    ax = fig.axes[0]
    # print('ax called')


    # 创建一个空的set
    label_set = set()

    x = df['SiO2(wt%)']
    y = df['Na2O(wt%)'] + df['K2O(wt%)']

    # 如果df中没有'Color'列，根据'label'生成颜色
    if 'Color' not in df.columns:
        if 'Label' in df.columns:
            unique_labels = df['Label'].unique()
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            color_dict = dict(zip(unique_labels, colors))
            df['Color'] = df['Label'].map(color_dict)
        else:
            df['Color'] = 'b'  # 默认颜色为蓝色
    
    # 如果df中没有'Marker'列，根据'label'生成符号
    if 'Marker' not in df.columns:
        if 'Label' in df.columns:
            unique_labels = df['Label'].unique()
            markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']  # 可用的matplotlib标记
            marker_dict = dict(zip(unique_labels, markers * len(unique_labels)))  # 如果标签数量超过标记类型，会循环使用标记
            df['Marker'] = df['Label'].map(marker_dict)
        else:
            df['Marker'] = 'o'  # 默认符号为圆圈

    color = df['Color']
    marker = df['Marker']
    alpha = df['Alpha'] if 'Alpha' in df.columns else 0.8
    size = df['Size'] if 'Size' in df.columns else 80
    label = df['Label'] 

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

    def plot_group(group):
        # ax.scatter(group['x'], group['y'], c=group['color'], alpha=group['alpha'], s=group['size'], label=group.name,edgecolors='black')
        ax.scatter(group['x'], group['y'], c=group['color'], alpha=0.3, s=group['size'], label=group.name)

    # 创建一个新的DataFrame，包含所有需要的列
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'color': color,
        'alpha': alpha,
        'size': size,
        'marker': marker,
        'label': label
    })

    # 按照'label'列进行分组，然后对每个组应用plot_group函数
    df.groupby('label').apply(plot_group)
    


    ax.legend()
    # Print the size of the figure
    # print('Figure size:', fig.get_size_inches())

    fig.dpi=dpi
    # 设置fig的尺寸
    fig.set_size_inches(16 , 16)
    fig.savefig('TAS_'+tag+'_'+file_name+'.svg', format='svg', dpi=dpi*10)
    fig.savefig('TAS_'+tag+'_'+file_name+'.jpg', format='jpg', dpi=dpi*10)

def export_result(file_name = "Geochemistry.csv"):   
    # 设置标签为'VOL'
    tag = 'VOL'
    
    # 设置设置为'Withlines'
    setting = 'Withlines'
    
    # 设置颜色设置为空字符串
    color_setting = ''
    
    # 设置数据路径为空字符串
    data_path = ''
    dpi = 100

    if file_name:
        # 根据文件扩展名判断并读取数据
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_name)  # 从CSV文件中读取数据
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_name)  # 从Excel文件中读取数据
        
    x = df['SiO2(wt%)']
    y = df['Na2O(wt%)'] + df['K2O(wt%)']            
    # 创建一个函数来判断一个点是否在一个多边形内
    def point_in_polygon(point, polygon):
        return Path(polygon).contains_points([point])

    # 创建一个列表来保存所有的标签
    type_list = []
    Polygon_dict, tas_cord = generate_polygon()
    # 遍历x和y坐标数组中的所有点对
    for x_val, y_val in zip(x, y):
        # 对于每个点，遍历字典Polygon_dict中存储的所有多边形及其类型标签
        for type_label, polygon in Polygon_dict.items():
            # 判断当前点是否位于该多边形内
            if point_in_polygon((x_val, y_val), polygon.get_xy()):
                # 若点在多边形内，则从tas_cord字典的"Volcanic"键下获取与type_label对应的值，并添加到type_list列表中
                type_list.append(tas_cord["Volcanic"].get(type_label))
                # 找到符合条件的第一个多边形后跳出内层循环
                break
        # 若点不在任何多边形内，则将None添加到type_list列表中
        else:
            type_list.append(None)

    # 将type_list内容转换为DataFrame，设置列名为'TAS as VOL'
    tas_df = pd.DataFrame({'TAS as VOL': type_list})

    # 设置文件路径，根据tag变量拼接字符串得到 '_GMS_kde' 后缀的文件名部分
    file_path = tag + '_GMS_kde'

    # Check if the path exists
    # 检查文件路径是否存在
    if os.path.exists(file_path):
        # 遍历目录下的所有文件
        kde_result = {}
        kde_result_divided_max = {}

        # 将x和y数据合并为二维数组data_test
        data_test = np.column_stack((x, y))

        # 遍历指定目录中的每个文件名
        for filename in os.listdir(file_path):
            # 提取文件名中表示目标类型的字符串（去掉"_kde.pkl"后缀）
            type_target = filename.replace('_kde.pkl', '')
            # 构造完整文件路径
            full_file_path = os.path.join(file_path, filename)

            # 以二进制模式打开并加载该文件中的pickle对象（即KDE模型）
            with open(full_file_path, 'rb') as f:
                kde = pickle.load(f)

            # 创建一个用于计算概率密度的网格数据，横纵坐标范围分别为[35, 90]和[0, 20]
            data_whole = np.column_stack((np.linspace(35, 90, 1024), np.linspace(0, 20, 1024)))

            # 计算训练集每个点对应的对数概率密度值
            log_whole_prob_density = kde.score_samples(data_whole)
            # 将对数概率密度转换为原始概率密度
            prob_density = np.exp(log_whole_prob_density)

            # 使用KDE模型计算测试数据集的对数概率密度值
            test_densities = kde.score_samples(data_test)

            # 将测试数据集的对数概率密度转换为原始概率密度
            test_probabilities = np.exp(test_densities)

            # 存储每种类型的目标对应的测试概率到字典kde_result中
            kde_result[type_target] = test_probabilities.round(3)

            # 计算训练集和测试集中的最大概率密度，取两者中较大的一个作为归一化基准
            max_prob_density = max(np.max(prob_density), np.max(test_probabilities))

            # 对测试数据集的概率密度进行归一化处理，使得所有概率密度值在0到1之间
            test_probabilities /= max_prob_density

            # 这里先最大值归一化，然后比值得到概率值，留待后续作SoftMax函数处理                    
            kde_result_divided_max[type_target] = test_probabilities.round(3)

        # 将字典kde_result转化为DataFrame格式
        kde_result_df = pd.DataFrame(kde_result)
        kde_result_divided_max_df = pd.DataFrame(kde_result_divided_max)
        # 定义归一化函数

        # Min-Max Scaling
        def min_max_scaling(x):
            return (x - np.min(x)) / (np.max(x) - np.min(x))

        # Z-score Normalization
        def z_score_normalization(x):
            return (x - np.mean(x)) / np.std(x)

        # Decimal Scaling
        def decimal_scaling(x):
            max_abs_val = np.max(np.abs(x))
            num_digits = np.floor(np.log10(max_abs_val) + 1)
            return x / (10 ** num_digits)

        # Softmax Scaling
        def softmax_scaling(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        # Simple Feature Scaling
        def simple_feature_scaling(x):
            return x / np.max(x)

        # Probability Proportional Scaling
        def probability_proportional_scaling(x):
            return x / np.sum(x)
        

        # 指数函数
        def exp_normalize(x):
            e_x = np.exp(x)
            return e_x / np.sum(e_x)

        # 幂函数
        def power_normalize(x, power):
            p_x = np.power(x, power)
            return p_x / np.sum(p_x)

        # 对DataFrame的每一行应用归一化函数
        kde_result_df = kde_result_df.apply(softmax_scaling, axis=1)

        # kde_result_divided_max_df = kde_result_divided_max_df.apply(probability_proportional_scaling, axis=1)

        # print(kde_result_df)

        # 创建新的DataFrame来存储分类结果（最高概率对应的目标类型）
        kde_Type_df = pd.DataFrame(kde_result_df.idxmax(axis=1), columns=['Soft-Max Classification'])

        # 创建新的DataFrame来存储最大概率值
        kde_Probs_df = pd.DataFrame(kde_result_df.max(axis=1).round(6), columns=['Soft-Max Probability'])


        # 创建新的DataFrame来存储分类结果（最高概率对应的目标类型）
        kde_Type_divided_max_df = pd.DataFrame(kde_result_divided_max_df.idxmax(axis=1), columns=['Max_Ratio Classification'])

        # 创建新的DataFrame来存储最大概率值
        kde_Probs_divided_max_df = pd.DataFrame(kde_result_divided_max_df.max(axis=1).round(6), columns=['Max_Ratio Probability'])

        # 将分类结果DataFrame、概率最大值DataFrame以及其它两个预先存在的DataFrame tas_df和df沿列方向拼接在一起
        new_df = pd.concat([kde_Type_divided_max_df,kde_Probs_divided_max_df, kde_Type_df, kde_Probs_df, tas_df, df], axis=1)

    df.to_csv(file_name+"_TAS_Result.csv", index=True)



for i in range(100):
    # plot_data(file_name = f'sampled_TAS_data_{i}.csv')
    export_result(file_name = f'sampled_TAS_data_{i}.csv')