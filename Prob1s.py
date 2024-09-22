import platform
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import *
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

sns.set_context('notebook')
sns.set_theme(style="ticks", palette="pastel")
if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
else:  # Mac or other systems
    plt.rcParams['font.family'] = ['Arial Unicode MS']

# preprocess and load data
filepath = 'train_data.pkl'
if not os.path.exists(filepath):
    general_process(output_filepath=filepath)
with open(filepath, 'rb') as fin:
    df = pickle.load(fin)

# normalize all waveforms to the range [-1, 1] for better classification
df['flux_density'] = df['flux_density'].apply(
    lambda x: 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1)

def break_counter(sequence, step=25, eps=0.05, verbose=False):
    """ count the number of the "change" 
    of the first_derivative is bigger than eps """
    first_derivative = np.diff(sequence[::step])
    # second_derivative = np.diff(first_derivative)
    pivots = []
    for j in range(len(first_derivative) - 1):
        if np.abs(first_derivative[j+1] - first_derivative[j]) > eps:
            if not pivots or j - pivots[-1] > 2:
                pivots.append(j)
    if verbose:
        print(pivots)
    return len(pivots)

def evaluate_thresholds(thresholds, step=25):
    accuracies = []
    
    for eps in thresholds:
        correct_count = 0
        total_count = 0
        
        for _ in range(10_000):
            # 评估正弦波
            sequence = df[df['type_waveform'] == '正弦波'].sample(1)['flux_density'].values[0]
            if break_counter(sequence, step=step, eps=eps) == 0:
                correct_count += 1
            
            # 评估三角波
            record = df[df['type_waveform'] == '三角波'].sample(1)
            sequence = record.iloc[0, -1]
            break_count = break_counter(sequence, step=step, eps=eps)
            if break_count in [1, 2]:
                correct_count += 1
            
            # 评估梯形波
            record = df[df['type_waveform'] == '梯形波'].sample(1)
            sequence = record.iloc[0, -1]
            break_count = break_counter(sequence, step=step, eps=eps)
            if break_count >= 3:
                correct_count += 1
            
            total_count += 3  # 每个波形类型各计一次
        
        # 计算准确度
        accuracy = correct_count / total_count
        accuracies.append(accuracy)
    
    return accuracies

# 定义不同的阈值
thresholds = np.arange(0.01, 0.2, 0.01)
accuracies = evaluate_thresholds(thresholds)

# 绘制准确度与阈值的关系图
plt.figure(figsize=(10, 5))
plt.plot(thresholds, accuracies, marker='o')
plt.xlabel('阈值')
plt.ylabel('分类准确度')
plt.grid()
plt.xticks(thresholds, rotation=45)
plt.show()