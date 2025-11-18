# 导入工具包
from sklearn.datasets import load_iris          # 加载鸢尾花测试集的.
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    # 分割训练集和测试集的
from sklearn.preprocessing import StandardScaler        # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier      # KNN算法 分类对象
from sklearn.metrics import accuracy_score              # 模型评估的, 计算模型预测的准确率

def pre_iris():
    # 1.获取数据
    iris_data = load_iris()
    # 2.数据预处理
        # 只需要进行训练数据和测试数据的切分
    x_train,x_test,y_train,y_test =train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=21)
    print(f'x_train: {x_train[:5]}')
    # print(f'y_train: {y_train[:5]}')
    # print(f'训练数据数量: {len(x_train),len((y_train))}')
    # print(f'x_test: {x_test}')
    # print(f'y_test: {y_test}')
    # 3.特征工程
        # 3.1 特征提取 特征预处理（归一化 标准化） 特征降维 特征选取 特征组合、
    standardizer = StandardScaler()
    x_train = standardizer.fit_transform(x_train)
    x_test = standardizer.transform(x_test)

    # 4.模型训练
    knn_model=KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(x_train,y_train)

    # 5.模型预测
    y_pre = knn_model.predict(x_test)
    print(f'y_pred: {y_pre}')
    print(f'y_test: {y_test}')

    # 针对一个新数据 进行评估
    x_test_new = [[2.2,1.5,2.0,1.6]]
    x_test_new = standardizer.transform(x_test_new)

    y_pre_new = knn_model.predict_proba(x_test_new)
    print(f'y_pre_new: {y_pre_new}')
    # 6.模型评估
    print(accuracy_score(y_test, y_pre))

if __name__ == '__main__':
    pass
    # pre_iris()