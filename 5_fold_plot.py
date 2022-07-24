from matplotlib import pyplot as plt
from numpy import interp
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import classify_parameter

if __name__ == '__main__':
    # 获取数据集及标签,打乱数据(标签有序或过于集中会导致交叉验证时,只有一种样本,导致roc的area为nan)
    # 解决RuntimeWarning: invalid value encountered in true_divide
    np.seterr(divide='ignore', invalid='ignore')
    dataframe = shuffle(pd.read_csv('result/fill_with_average.csv', encoding='utf-8'))
    ards = dataframe.iloc[:, 1:-5].values
    label = dataframe.iloc[:, -5].values
    # print(str(ards) + ' label is : ' + str(label))
    label_new = []
    # 数据类型转换
    for item in label:
        if item == 1:
            label_new.append(1)
        else:
            label_new.append(0)
    label_new = np.array(label_new)
    # X_train, X_test, y_train, y_test = train_test_split(ards, label_new, test_size=0.2, random_state=123)
    # normalize train
    # X_train = MinMaxScaler().fit_transform(X_train)
    # normalize test
    # X_test = MinMaxScaler().fit_transform(X_test)
    # 设置五折交叉检验
    KF = KFold(n_splits=5)
    tprs = []
    aucs = []
    # 格式： array = numpy.linspace(start, end, num=num_points)
    # 将在start和end之间生成一个统一的序列，共有num_points个元素
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    model = classify_parameter.GBDT_GridSearchCV
    # 五折交叉检验
    for train_index, test_index in KF.split(ards):
        # 划分训练集和测试集
        x_train, x_test = ards[train_index], ards[test_index]
        y_train, y_test = label_new[train_index], label_new[test_index]
        # 模型训练
        model.fit(x_train, y_train)
        # 模型预测
        test_predict_proba = model.predict_proba(x_test)
        train_predict_proba = model.predict_proba(x_train)
        print('The train accuracy of the GBDT is:', metrics.accuracy_score(y_train, train_predict_proba))
        print('The test accuracy of the GBDT is:', metrics.accuracy_score(y_test, test_predict_proba))
        fpr, tpr, threshold = metrics.roc_curve(y_test, test_predict_proba[:, 1], pos_label=1)
        # 模型准确率计算
        roc_auc = metrics.auc(fpr, tpr)
        # 一维线性插值
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0
        aucs.append(auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.2f)' % (i, roc_auc))
        i = i + 1
    # 画对角线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
    std_auc = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.2f)' % mean_auc, lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()
