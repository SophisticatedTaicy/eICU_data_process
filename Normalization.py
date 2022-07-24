import time
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.metrics import r2_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import classify_parameter


def normalization(data):
    # 提取指定特征数据
    feature_data = data.iloc[:, 1:-5]
    label = data.iloc[:, -5]
    # print(str(label))
    # print(str(feature_data))
    # 计算原始数据每行和每列的均值和方差，feature_data是多维数据
    min_max = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(feature_data)
    # z_score = preprocessing.scale(feature_data)
    # k = np.ceil(np.log10(np.max(abs(feature_data))))
    # decimal_scaling = feature_data / (10 ** k)
    data_normal = DataFrame(min_max)
    data_normal.insert(loc=len(data_normal.columns), column='status', value=label)
    data_normal.to_csv('result/data_normal.csv', mode='w', encoding='utf-8', index=False)


if __name__ == '__main__':
    start = time.time()
    print('开始时间 ： ' + str(datetime.now()))
    data_csv = pd.read_csv('result/fill_with_average.csv', sep=',')
    # 'utf-8' codec can't decode byte 0xb0 in position 113182: invalid start byte 可能文件中含有中文或者其他不支持字符
    normalization(data_csv)
    data = shuffle(pd.read_csv('result/data_normal.csv', encoding='utf-8'))
    ards = np.array(data.iloc[:, 1:-1])
    label = np.array(data.iloc[:, -1])
    # print(str(ards) + ' label is : ' + str(label))
    label_new = []
    # 数据类型转换
    for item in label:
        if item == 1:
            label_new.append(1)
        else:
            label_new.append(0)
    label_new = np.array(label_new)
    # print('label is : ' + str(label_new))
    x_train, x_test, y_train, y_test = train_test_split(ards, label_new, test_size=0.2)
    end = time.time()
    total = (end - start) / 60
    print('结束时间 ： ' + str(datetime.now()) + '程序运行花费了' + str(time.time() - start) + '秒')
