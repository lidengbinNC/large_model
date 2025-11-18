# 导入工具包
from sklearn.datasets import load_iris          # 加载鸢尾花测试集的.
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    # 分割训练集和测试集的
from sklearn.preprocessing import StandardScaler        # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier      # KNN算法 分类对象
from sklearn.metrics import accuracy_score              # 模型评估的, 计算模型预测的准确率

def view_digits(idx):
    df = pd.read_csv('./data/手写数字识别.csv')
    print(f'df.shape: {df.shape}')
    x = df.iloc[:,1:]
    y = df.iloc[:,0]
    print(f'x: {x}')
    print(f'y: {y}')

if __name__ == '__main__':
    view_digits(
        8)