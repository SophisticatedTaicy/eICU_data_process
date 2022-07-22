from matplotlib import pyplot as plt
from numpy import interp
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import model_parameter

if __name__ == '__main__':
    # 获取数据集及标签,打乱数据(标签有序或过于集中会导致交叉验证时,只有一种样本,导致roc的area为nan)
    # dataframe = shuffle(pd.read_csv('data/new_result.csv', encoding='utf-8', low_memory=False))
    dataframe = shuffle(pd.read_csv('result/fill_with_average.csv', encoding='utf-8'))
    ards = np.array(dataframe.iloc[:, 1:-1])
    # print(str(ards))
    label = np.array(dataframe.iloc[:, -1])
    # print(str(ards) + ' label is : ' + str(label))
    label_new = []
    # 数据类型转换
    for item in label:
        if item == 1:
            label_new.append(1)
        else:
            label_new.append(0)
    label_new = np.array(label_new)
    # 设置五折交叉检验
    KF = KFold(n_splits=5)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    model = GradientBoostingRegressor(random_state=123)
    # 五折交叉检验
    for train_index, test_index in KF.split(ards):
        x_train, x_test = ards[train_index], ards[test_index]
        y_train, y_test = label_new[train_index], label_new[test_index]
        # model = model_parameter.fold5_GBDT
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        # gbdt_classify_model = filter.param.classify_model
        # gbdt_classify_model.fit(x_train, y_train)
        # y_pred = gbdt_classify_model.predict(x_test)
        fpr, tpr, threhold = roc_curve(y_test, y_pred)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0
        roc_auc = auc(fpr, tpr)
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
