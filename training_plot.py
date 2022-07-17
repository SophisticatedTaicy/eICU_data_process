import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle

import filter.param

if __name__ == '__main__':
    # 获取数据集及标签,打乱数据(标签有序或过于集中会导致交叉验证时,只有一种样本,导致roc的area为nan)
    dataframe = shuffle(pd.read_csv('data/result.csv', encoding='utf-8'))
    ards = np.array(dataframe.iloc[:, 1:-1])
    label = np.array(dataframe.iloc[:, -1])
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
    i = 0
    x_train, x_test, y_train, y_test = train_test_split(ards, label_new, test_size=0.2)
    model = filter.param.reg_model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    fpr, tpr, threhold = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    aucs.append(auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='training (area=%0.2f)' % (roc_auc))
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
