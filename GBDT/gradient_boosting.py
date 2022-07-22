import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from IPython.display import Image
from pydotplus import graph_from_dot_data
from sklearn.model_selection import train_test_split
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.datasets import load_boston
from sklearn.tree import export_text
import filter.param
from sklearn.metrics import mean_squared_error

os.environ['PATH'] += os.pathsep + 'D:/graphviz/bin/'


# 提升数的形成和计算流程
class BoostingTree:
    def __init__(self, error=1e-2):
        self.error = error  # 误差值
        self.candidate_splits = []  # 候选切分点
        self.split_index = defaultdict(tuple)  # 由于要多次切分数据集，故预先存储，切分后数据点的索引
        self.split_list = []  # 最终各个基本回归树的切分点
        self.c1_list = []  # 切分点左区域取值（均值）
        self.c2_list = []  # 切分点右区域取值（均值）
        self.N = None  # 数组元素个数
        self.n_split = None  # 切分点个数

    def split_array(self, X_data):
        self.N = X_data.shape[0]
        # 提取数据的切分点,默认切分点为前后两个数取均值
        for i in range(1, self.N):
            self.candidate_splits.append((X_data[i][0] + X_data[i - 1][0]) / 2)
        self.n_split = len(self.candidate_splits)
        # 按照切分点将原数据一分为二
        for split in self.candidate_splits:
            left_index = np.where(X_data[:, 0] < split)[0]
            right_index = np.where(X_data[:, 0] > split)[0]
            self.split_index[split] = (left_index, right_index)
        return

    # split为切分点,计算各切分点误差
    def calculate_error(self, split, y_result):
        indexs = self.split_index[split]
        # 切分点左边数据
        left = y_result[indexs[0]]
        # 切分点右边数据
        right = y_result[indexs[1]]
        # 左边数据均值
        c1 = np.sum(left) / len(left)
        # 右边数据均值
        c2 = np.sum(right) / len(right)
        left_index_error = left - c1
        right_index_error = right - c2
        result = np.hstack([left_index_error, right_index_error])
        # error=(left-c1)^2+(right-c2)^2
        error = np.apply_along_axis(lambda x: x ** 2, 0, result).sum()
        return error, c1, c2

    # 选出最佳切分点，即使得误差最小的点
    def best_split(self, y_result):
        best_split = self.candidate_splits[0]
        best_error, best_c1, best_c2 = self.calculate_error(best_split, y_result)
        for i in range(1, self.n_split):
            error, c1, c2 = self.calculate_error(self.candidate_splits[i], y_result)
            if error < best_error:
                best_split = self.candidate_splits[i]
                best_error = error
                best_c2 = c2
                best_c1 = c1
        self.split_list.append(best_split)
        self.c1_list.append(best_c1)
        self.c2_list.append(best_c2)

    # 预测x的残差
    def predict_x(self, x):
        s = 0
        for split, c1, c2 in zip(self.split_list, self.c1_list, self.c2_list):
            if x < split:
                s += c1
            else:
                s += c2
        return s

    # 每添加一棵回归树，就更新y，基于当前组合回归树的残差预测
    def update_y(self, x_data, y_data):
        y_result = []
        for x, y in zip(x_data, y_data):
            y_result.append(y - self.predict_x(x[0]))
        y_result = np.array(y_result)
        error = np.apply_along_axis(lambda x: x ** 2, 0, y_result).sum()
        return y_result, error

    def fit(self, x_data, y_data):
        # 计算数据切分点
        self.split_array(x_data)
        y_result = y_data
        while True:
            # 找到数据最佳切分点
            self.best_split(y_result)
            y_result, error = self.update_y(x_data, y_data)
            # 残差小于指定阈值时结束
            if error < self.error:
                break
        return

    def predict(self, x):
        return self.predict_x(x)


# 梯度提升回归模型使用
def gradient_boosting():
    x = np.arange(1, 11).reshape(-1, 1)
    y = np.array([5.16, 4.73, 5.95, 6.42, 6.88, 7.15, 8.95, 8.71, 9.50, 9.15])
    gbdt = GradientBoostingRegressor(max_depth=4, criterion='squared_error').fit(x, y)
    sub_tree = gbdt.estimators_[4, 0]
    dot_data = export_graphviz(sub_tree, out_file=None, filled=True, rounded=True, special_characters=True, precision=2)
    graph = graph_from_dot_data(dot_data)
    Image(graph.create_png())


# 梯度提升分类模型使用
def gbdt():
    '''
    调参：
    loss：损失函数。有log_loss和exponential两种。deviance是采用对数似然，exponential是指数损失，后者相当于AdaBoost。
    n_estimators:最大弱学习器个数，默认是100，调参时要注意过拟合或欠拟合，一般和learning_rate一起考虑。
    learning_rate:步长，即每个弱学习器的权重缩减系数，默认为0.1，取值范围0-1，当取值为1时，相当于权重不缩减。较小的learning_rate相当于更多的迭代次数。
    subsample:子采样，默认为1，取值范围(0,1]，当取值为1时，相当于没有采样。小于1时，即进行采样，按比例采样得到的样本去构建弱学习器。这样做可以防止过拟合，但是值不能太低，会造成高方差。
    init：初始化弱学习器。不使用的话就是第一轮迭代构建的弱学习器.如果没有先验的话就可以不用管
    由于GBDT使用CART回归决策树。以下参数用于调优弱学习器，主要都是为了防止过拟合
    max_feature：树分裂时考虑的最大特征数，默认为None，也就是考虑所有特征。可以取值有：log2,auto,sqrt
    max_depth：CART最大深度，默认为None
    min_sample_split：划分节点时需要保留的样本数。当某节点的样本数小于某个值时，就当做叶子节点，不允许再分裂。默认是2
    min_sample_leaf：叶子节点最少样本数。如果某个叶子节点数量少于某个值，会同它的兄弟节点一起被剪枝。默认是1
    min_weight_fraction_leaf：叶子节点最小的样本权重和。如果小于某个值，会同它的兄弟节点一起被剪枝。一般用于权重变化的样本。默认是0
    min_leaf_nodes：最大叶子节点数
    '''

    gbdt = GradientBoostingClassifier(loss='log_loss', learning_rate=1, n_estimators=5, subsample=1,
                                      min_samples_split=2, min_samples_leaf=1, max_depth=2, init=None,
                                      random_state=None, max_features=None, verbose=0, max_leaf_nodes=None,
                                      warm_start=False)

    train_feat = np.array([[6], [12], [14], [18], [20], [65], [31], [40], [1], [2], [100], [101], [65], [54]])
    train_label = np.array([[0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [2], [2], [2], [2]]).ravel()
    test_feat = np.array([[25]])
    test_label = np.array([[0]])
    print(train_feat.shape, train_label.shape, test_feat.shape, test_label.shape)
    gbdt.fit(train_feat, train_label)
    pred = gbdt.predict(test_feat)
    print(pred, test_label)


def GBDT_Regressor_demo(data, label):
    # 参考
    # https: // blog.csdn.net / bitcarmanlee / article / details / 77857194
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3)

    filter.param.reg_model.fit(x_train, y_train)

    # 回归
    prediction_train = filter.param.reg_model.predict(x_train)
    rmse_train = mean_squared_error(y_train, prediction_train)

    prediction_test = filter.param.reg_model.predict(x_test)
    rmse_test = mean_squared_error(y_test, prediction_test)

    print("RMSE for training dataset is %f, for testing dataset is %f." % (rmse_train, rmse_test))


def GBDT_Classifier_demo(data, label):
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

    filter.param.classify_model.fit(x_train, y_train)

    # 回归
    prediction_train = filter.param.reg_model.predict(x_train)
    rmse_train = mean_squared_error(y_train, prediction_train)

    prediction_test = filter.param.reg_model.predict(x_test)
    rmse_test = mean_squared_error(y_test, prediction_test)

    print("RMSE for training dataset is %f, for testing dataset is %f." % (rmse_train, rmse_test))


if __name__ == '__main__':
    ards = pd.read_csv('../data/admission_score_result_0.csv', encoding='utf-8').iloc[:, 1:-1]
    label = list(pd.read_csv('../data/admission_score_result_0.csv', encoding='utf-8')['status'])
    label_new = []
    for item in label:
        if item > 1:
            label_new.append(2)
        else:
            label_new.append(item)
    # 15000条200维，三种结果，epoch20即收敛
    # GBDT_Regressor_demo(ards, label_new)
    # 1000条10维,200epoch未收敛
    # data_url = "http://lib.stat.cmu.edu/datasets/boston"
    # raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    # data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    # target = raw_df.values[1::2, 2]
    # GBDT_DEMO(data, target)

    # GBDT_Classifier_demo(ards, label_new)

    # 分类
    GBDTreg = GradientBoostingClassifier(**filter.param.params)
    GBDTreg.fit(ards, label)
    for i in range(0, GBDTreg.n_estimators):
        for j in range(0, 3):
            sub_tree = GBDTreg.estimators_[i, j]
            plt.figure(figsize=(15, 9))
            plot_tree(sub_tree)
            r1 = export_graphviz(sub_tree)
            print(r1)
            r2 = export_text(sub_tree, feature_names=filter.param.result_header[1:])
            print(r2)
    y_predict = GBDTreg.predict(ards)

    # gradient_boosting()
    # gbdt()
    # 提升树
    # data = np.array([[1, 5.56], [2, 5.70], [3, 5.91], [4, 6.40], [5, 6.80],
    #                  [6, 7.05], [7, 8.90], [8, 8.70], [9, 9.00], [10, 9.05]])
    # X_data = data[:, :-1]
    # y_data = data[:, -1]
    # bt = BoostingTree(error=0.18)
    # bt.fit(X_data, y_data)
    # print('切分点：', bt.split_list)
    # print('切分点左区域取值:', np.round(bt.c1_list, 2))
    # print('切分点右区域取值:', np.round(bt.c2_list, 2))
    # X_data_raw = np.linspace(-5, 5, 100)
    # X_data = np.transpose([X_data_raw])
    # y_data = np.sin(X_data_raw)
    # BT = BoostingTree(error=0.1)
    # BT.fit(X_data, y_data)
    # y_pred = [BT.predict(X) for X in X_data]
    #
    # p1 = plt.scatter(X_data_raw, y_data, color='r')
    # p2 = plt.scatter(X_data_raw, y_pred, color='b')
    # plt.legend([p1, p2], ['real', 'pred'])
    # plt.show()
