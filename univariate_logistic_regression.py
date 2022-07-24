import numpy as np
import copy

import os
# os.environ['PATH'] = os.pathsep + 'C:\\Program Files\\Graphviz 2.44.1\\bin'
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class Node:
    """
    树的节点类
    """

    def __init__(self, feature=-1, split_val=None, results=None, left=None, right=None):
        """
        :param feature: 用于切分数据集的特征索引
        :param split_val: 设置切分的值
        :param results: 存储节点的值
        :param left: 左子树
        :param right: 右子树
        """

        self.feature = feature
        self.split_val = split_val
        self.results = results
        self.left = left
        self.right = right


'''
说明：为了便于实现，dataSet类型ndarray，
dataSet: [X,Y,y_res]  #dataSet的组成结构
X:样本训练集
Y:样本标签
y_res: 残差
'''


def leaf(dataSet):
    """计算节点的数值
    :param dataSet: {ndarray}训练样本
    :return: 均值
    """
    '''
    生成叶子节点
    '''
    return np.sum(dataSet[:, -1]) / (np.sum((dataSet[:, -2] - dataSet[:, -1]) * (1 - dataSet[:, -2] + dataSet[:, -1])))


def err_cnt(dataSet):
    """计算误差
    :param dataSet: {ndarray}训练数据
    :return: 总方差
    """

    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def split_tree(dataSet, feature, split_val):
    """根据特征feature中的值split_val将数据集data划分为左右子树
    :param data: {list}训练样本
    :param feature: {int}需要划分的特征索引
    :param split_val: {float}指定的划分值
    :return:(set_1, set_2): {tuple} 左右子树的集合
    """

    set_L = dataSet[np.nonzero(dataSet[:, feature] <= split_val)[0], :]
    set_R = dataSet[np.nonzero(dataSet[:, feature] > split_val)[0], :]
    return set_L, set_R


class CART_regression(object):
    """
    CART算法类
    """

    def __init__(self, X, Y, min_sample, min_err, max_height=20):
        """
        :param X: 回归样本数据的特征
        :param Y: 回归样本数据的标签
        :param min_sample: 每个叶节点最少样本数
        :param min_err: 最小损失
        """
        self.X = X
        self.Y = Y
        self.min_sample = min_sample
        self.min_err = min_err
        self.max_height = max_height

    def fit(self):
        """
        构建树
        input:data{list} -- 训练样本
              min_sample{int} -- 叶子节点中最少样本数
              min_err{float} -- 最小的error
        output: node:树的根节点
        """
        # 将样本特征与样本标签合成完整的样本
        # X存放带样本标签的数据集，Y存放第i次的残差
        data = np.c_[self.X, self.Y]
        # 初始化
        best_err = err_cnt(data)
        # 存储最佳切分属性及最佳切分点
        bestCriteria = None
        # 存储切分后的两个数据集
        bestSets = None
        # 构建决策树，返回该决策树的根节点
        if np.shape(data)[0] <= self.min_sample or self.max_height == 1 or best_err <= self.min_err:
            return Node(results=leaf(data))

        # 开始构建CART回归树
        num_feature = np.shape(data[0])[0] - 2
        for feat in range(num_feature):
            val_feat = np.unique(data[:, feat])
            for val in val_feat:
                # 尝试划分
                set_L, set_R = split_tree(data, feat, val)
                if np.shape(set_L)[0] < 2 or np.shape(set_R)[0] < 2:
                    continue
                # 计算划分后的error值
                err_now = err_cnt(set_L) + err_cnt(set_R)
                # 更新最新划分
                if err_now < best_err:
                    best_err = err_now
                    bestCriteria = (feat, val)
                    bestSets = (set_L, set_R)
        # 生成左右子树
        left = CART_regression(bestSets[0][:, :-1], bestSets[0][:, -1], self.min_sample, self.min_err,
                               self.max_height - 1).fit()
        right = CART_regression(bestSets[1][:, :-1], bestSets[1][:, -1], self.min_sample, self.min_err,
                                self.max_height - 1).fit()
        return Node(feature=bestCriteria[0], split_val=bestCriteria[1], left=left, right=right)


def predict(sample, tree):
    f"""对每一个样本sample进行预测
    :param sample: {list}:样本
    :param tree: 训练好的CART回归模型
    :return: results{float} :预测值
    """

    # 叶子节点
    if tree.results is not None:
        return tree.results
    else:
        # 不是叶节点
        val_sample = sample[tree.feature]
        branch = None
        # 选择右子树
        if val_sample > tree.split_val:
            branch = tree.right
        else:
            branch = tree.left
        return predict(sample, branch)


def test(X, tree):
    """评估CART回归模型
    :param X: {list} 测试样本
    :param Y: {list} 测试标签
    :param tree: 训练好的CART回归树模型
    :return:  均方误差
    """

    m = np.shape(X)[0]
    y_hat = []
    for i in range(m):
        pre = predict(X[i], tree)
        y_hat.append(pre)
    return y_hat


def numLeaf(tree):
    if tree.results is not None:
        return 1
    else:
        return numLeaf(tree.left) + numLeaf(tree.right)


def heightTree(tree):
    if tree.results is not None:
        return 1
    else:
        heightL = heightTree(tree.left)
        heughtR = heightTree(tree.right)
        if heightL > heughtR:
            return heightL + 1
        else:
            return heughtR + 1


def showTree(tree):
    node = {}

    if tree.results is None:
        node['feat'] = tree.feature
        node['splitVal'] = tree.split_val
        print(node)
        showTree(tree.left)
        showTree(tree.right)
    else:
        node['value'] = tree.results
        print(node)


def load_data(data_file):
    """导入训练数据
    :param data_file: {string} 保存训练数据的文件
    :return: {list} 训练数据
    """
    X, Y = [], []
    f = open(data_file)
    for line in f.readlines():
        sample = []
        lines = line.strip().split('\t')
        Y.append(lines[-1])
        for i in range(len(lines) - 1):
            sample.append(float(lines[i]))
        X.append(sample)
    return X, Y


# 二分类输出非线性映射
def sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


class GBDT_RT(object):
    """
    GBDT回归算法类
    """

    def __init__(self):
        self.trees = None
        self.learn_rate = None
        self.init_val = None

    def get_init_value(self, y):
        """计算初始值的平均值
        :param y: {ndarray} 样本标签列表
        :return: average:{float} 样本标签的平均值
        """
        """
        初始化：
        F0(x)=log(p1/(1 - p1))
        """
        p = np.count_nonzero(y)
        n = np.shape(y)[0]
        return np.log(p / (n - p))

    def get_residuals(self, y, y_hat):
        """
        计算样本标签域预测列表的残差
        :param y: {ndarray} 样本标签列表
        :param y_hat: {ndarray} 预测标签列表
        :return: y_residuals {list} 样本标签与预测标签列表的残差
        """

        y_residuals = []
        for i in range(len(y)):
            y_residuals.append(y[i] - y_hat[i])
        return y_residuals

    def fit(self, X, Y, n_estimates, learn_rate, min_sample, min_err, max_height):
        """
        训练GDBT模型
        :param X: {list} 样本特征
        :param Y: {list} 样本标签
        :param n_estimates: {int} GBDT中CART树的个数
        :param learn_rate: {float} 学习率
        :param min_sample: {int} 学习CART时叶节点最小样本数
        :param min_err: {float} 学习CART时最小方差
        """

        # 初始化预测标签和残差
        self.init_val = self.get_init_value(Y)

        n = np.shape(Y)[0]
        F = np.array([self.init_val] * n)
        y_hat = np.array([sigmoid(self.init_val)] * n)
        y_residuals = Y - y_hat
        y_residuals = np.c_[Y, y_residuals]

        self.trees = []
        self.learn_rate = learn_rate
        # 迭代训练GBDT
        for j in range(n_estimates):
            tree = CART_regression(X, y_residuals, min_sample, min_err, max_height).fit()
            for k in range(n):
                res_hat = predict(X[k], tree)
                # 计算此时的预测值等于原预测值加残差预测值
                F[k] += self.learn_rate * res_hat
                y_hat[k] = sigmoid(F[k])
            y_residuals = Y - y_hat
            y_residuals = np.c_[Y, y_residuals]
            self.trees.append(tree)

    def GBDT_predicts(self, X_test):
        """
        预测多个样本
        :param X_test: {list} 测试集
        :return: predicts {list} 预测的结果
        """
        predicts = []
        for i in range(np.shape(X_test)[0]):
            pre_y = self.init_val
            for tree in self.trees:
                pre_y += self.learn_rate * predict(X_test[i], tree)
            if sigmoid(pre_y) >= 0.5:
                predicts.append(1)
            else:
                predicts.append(0)
        return predicts

    def cal_error(self, Y_test, predicts):
        """
        计算预测误差
        :param Y_test: {测试样本标签列表}
        :param predicts: {list} 测试样本预测列表
        :return: error {float} 均方误差
        """

        y_test = np.array(Y_test)
        y_predicts = np.array(predicts)
        error = np.square(y_test - y_predicts).sum() / len(Y_test)
        return error


if __name__ == '__main__':
    # name of features
    # read data file
    data = pd.read_csv('result/fill_with_average.csv', sep=',')
    # set random seed
    np.random.seed(123)
    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 1:-5].values, data.iloc[:, -5].values,
                                                   test_size=0.2, random_state=123)
    # normalize train
    # X_train = MinMaxScaler().fit_transform(X_train)
    # normalize test
    # X_test = MinMaxScaler().fit_transform(X_test)
    gbdtTrees = GBDT_RT()
    gbdtTrees.fit(X_train, y_train)
    for i in range(2):
        print(numLeaf(gbdtTrees.trees[i]))
        print(heightTree(gbdtTrees.trees[i]))
        showTree(gbdtTrees.trees[i])
        print('--------------------------------------------')
    y_hat = gbdtTrees.GBDT_predicts(X_test)
    print(y_hat)
    acc = accuracy_score(y_test, y_hat)
    print(acc)
