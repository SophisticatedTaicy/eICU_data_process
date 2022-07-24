import numpy as np
import pandas as pd
import sns as sns
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score, accuracy_score, recall_score, \
    precision_score, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle
import seaborn as sns
import model_parameter


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


if __name__ == '__main__':
    # 获取数据集及标签,打乱数据(标签有序或过于集中会导致交叉验证时,只有一种样本,导致roc的area为nan)
    # 未归一化数据
    # dataframe = shuffle(pd.read_csv('data/new_result.csv', encoding='utf-8'))
    # 归一化后数据
    np.seterr(divide='ignore', invalid='ignore')
    dataframe = shuffle(pd.read_csv('result/data_normal.csv', encoding='utf-8'))
    # dataframe = shuffle(pd.read_csv('result/fill_with_0.csv', encoding='utf-8', low_memory=False))
    ards = np.array(dataframe.iloc[:, 1:-1])
    # 查看含有-的位置
    # print(str(np.dstack((np.where(ards == '-')[0], np.where(ards == '-')[1])).squeeze()))
    label = np.array(dataframe.iloc[:, -1])
    label_new = []
    # 数据类型转换
    for item in label:
        if item == 1:
            label_new.append(1)
        else:
            label_new.append(0)
    label_new = np.array(label_new)
    tprs = []
    aucs = []
    i = 0
    print('ards is : ' + str(ards) + ' label is : ' + str(label_new))
    x_train, x_test, y_train, y_test = train_test_split(ards, label, test_size=0.2)
    model = model_parameter.GBR
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    fpr, tpr, threhold = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    # aucs.append(auc)
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
