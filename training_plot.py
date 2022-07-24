import numpy as np
import pandas as pd
import sns as sns
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score, accuracy_score, recall_score, \
    precision_score, r2_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import seaborn as sns

import ards_classification
import classify_parameter
import matplotlib as mpl


def calculate_metrics(gt, pred):
    """
    :param gt: 数据的真实标签，一般对应二分类的整数形式，例如:y=[1,0,1,0,1]
    :param pred: 输入数据的预测值，因为计算混淆矩阵的时候，内容必须是整数，所以对于float的值，应该先调整为整数
    :return: 返回相应的评估指标的值
    """
    """
        confusion_matrix(y_true,y_pred,labels,sample_weight,normalize)
        y_true:真实标签；
        y_pred:预测概率转化为标签；
        labels:用于标签重新排序或选择标签子集；
        sample_weight:样本权重；
        normalize:在真实（行）、预测（列）条件或所有总体上标准化混淆矩阵；
    """
    print("starting!!!-----------------------------------------------")
    sns.set()
    fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), nrows=2)
    confusion = confusion_matrix(gt, pred)
    # 打印具体的混淆矩阵的每个部分的值
    print(confusion)
    # 从左到右依次表示TN、FP、FN、TP
    print(confusion.ravel())
    # 绘制混淆矩阵的图
    sns.heatmap(confusion, annot=True, cmap='Blues', linewidths=0.5, ax=ax1)
    ax2.set_title('sns_heatmap_confusion_matrix')
    ax2.set_xlabel('y_pred')
    ax2.set_ylabel('y_true')
    fig.savefig('sns_heatmap_confusion_matrix.jpg', bbox_inches='tight')
    # 混淆矩阵的每个值的表示
    (TN, FP, FN, TP) = confusion.ravel()
    # 通过混淆矩阵计算每个评估指标的值
    print('AUC:', roc_auc_score(gt, pred))
    print('Accuracy:', (TP + TN) / float(TP + TN + FP + FN))
    print('Sensitivity:', TP / float(TP + FN))
    print('Specificity:', TN / float(TN + FP))
    print('PPV:', TP / float(TP + FP))
    print('Recall:', TP / float(TP + FN))
    print('Precision:', TP / float(TP + FP))
    # 用于计算F1-score = 2*recall*precision/recall+precision,这个情况是比较多的
    P = TP / float(TP + FP)
    R = TP / float(TP + FN)
    print('F1-score:', (2 * P * R) / (P + R))
    print('True Positive Rate:', round(TP / float(TP + FN)))
    print('False Positive Rate:', FP / float(FP + TN))
    print('Ending!!!------------------------------------------------------')

    # 采用sklearn提供的函数验证,用于对比混淆矩阵方法与这个方法的区别
    print("the result of sklearn package")
    auc = roc_auc_score(gt, pred)
    print("sklearn auc:", auc)
    accuracy = accuracy_score(gt, pred)
    print("sklearn accuracy:", accuracy)
    recal = recall_score(gt, pred)
    precision = precision_score(gt, pred)
    print("sklearn recall:{},precision:{}".format(recal, precision))
    print("sklearn F1-score:{}".format((2 * recal * precision) / (recal + precision)))


# 展示真实值和预测值
def plot_test_and_predict(y_test, y_pred):
    # 为了正常显示中文
    mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
    mpl.rcParams['font.size'] = 12  # 字体大小
    plt.plot(y_pred, label='预测预后')
    plt.plot(y_test, label='真实预后')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 获取数据集及标签,打乱数据(标签有序或过于集中会导致交叉验证时,只有一种样本,导致roc的area为nan)
    # 未归一化数据
    # 解决RuntimeWarning: invalid value encountered in true_divide
    np.seterr(divide='ignore', invalid='ignore')
    data = shuffle(pd.read_csv('result/fill_with_average.csv', sep=','))
    # split data into train and test
    ards = data.iloc[:, 1:-5].values
    label = data.iloc[:, -5].values
    label_new = []
    # 数据类型转换
    for item in label:
        if item == 1:
            label_new.append(1)
        else:
            label_new.append(0)
    label_new = np.array(label_new)
    X_train, X_test, y_train, y_test = train_test_split(ards, label_new, test_size=0.2, random_state=123)
    # normalize train
    X_train = MinMaxScaler().fit_transform(X_train)
    # normalize test
    X_test = MinMaxScaler().fit_transform(X_test)
    tprs = []
    aucs = []
    i = 0
    model = classify_parameter.GBDTclas
    model.fit(X_train, y_train)
    # y_test_pred = model.predict(X_test)
    # fpr, tpr, threshold = metrics.roc_curve(y_test, y_test_pred)
    y_test_predict_proba = model.predict_proba(X_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_test_predict_proba[:-1], pos_label=1)
    # 计算预测值与真实值之间的均方误差
    # mse = mean_squared_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_predict_proba)
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
    y_train_pred = model.predict(X_train)
    # print('The train accuracy of the GBDT is:', metrics.accuracy_score(y_train, y_train_pred))
    print('The test accuracy of the GBDT is:', metrics.accuracy_score(y_test, y_test_predict_proba))
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='training (area=%0.4f)' % (roc_auc))
    # 画对角线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    std_auc = np.std(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()
