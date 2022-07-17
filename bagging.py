import itertools
import string

import numpy as np
from matplotlib import pyplot as plt, gridspec
from mlxtend.plotting import plot_decision_regions
from numpy import interp
from sklearn import svm
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier
from sklearn.decomposition import PCA
from stacking import RANDOM_SEED


def train(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(
        'model : ' + str(model) + 'train accuracy : ' + str(train_accuracy) + ' test accuracy : ' + str(test_accuracy))
    # roc曲线绘制
    fpr, tpr, threshold = roc_curve(y_test, y_test_pred)
    roc_auc = auc(fpr, tpr)
    aucs.append(auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='bagging training (area=%0.2f)' % (roc_auc))
    # 画对角线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()


def txt_to_array(train_txt, test_txt):
    # 读入数据txt
    x_test = []
    x_train = []
    y_test = []
    y_train = []
    for line in test_txt:
        # 将每行开头和结尾的换行符去掉,将tab作为间隔符
        line_new = line.strip('\n').split('	')
        item_list = []
        for item in range(0, len(line_new) - 1):
            item_list.append(float(line_new[item]))
        x_test.append(item_list)
        y_test.append(int(line_new[-1]))
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    # print('x_test : ' + str(x_test) + ' y_test : ' + str(y_test))
    for line in train_txt:
        # 去掉小数点后多余的0
        line = line.replace('.000000', '')
        # 去掉开头结尾的换行符
        line_new = line.strip('\n').split('\t')
        item_list = []
        for item in range(0, len(line_new) - 1):
            item_list.append(float(line_new[item]))
        x_train.append(item_list)
        y_train.append(int(line_new[-1]))
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train, y_train, x_test, y_test


def blank_fill(train_txt, test_txt):
    # 读入数据txt
    x_test = []
    x_train = []
    y_test = []
    y_train = []
    for line in test_txt:
        # 将每行开头和结尾的换行符去掉,将tab作为间隔符
        line_new = line.strip('\n').split('	')
        item_list = []
        for item in range(0, len(line_new) - 1):
            if item == '?':
                item = 0
        x_test.append(list(line_new[:-1]))
        y_test.append(int(line_new[-1]))
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    print('x_test : ' + str(x_test) + ' y_test : ' + str(y_test))
    for line in train_txt:
        # 去掉小数点后多余的0
        # line = line.replace('.000000', '')
        # 去掉开头结尾的换行符
        line_new = line.strip('\n').split('\t')
        item_list = []
        for item in range(0, len(line_new) - 1):
            if item == '?':
                item = 0

    print('x_train : ' + str(x_train) + ' y_train : ' + str(y_train))
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train, y_train, x_test, y_test


def wine_classify():
    import pandas as pd
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                       'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines', 'Proline']
    df_wine['Class label'].value_counts()
    '''
    2    71
    1    59
    3    48
    Name: Class label, dtype: int64
    '''
    df_wine = df_wine[df_wine['Class label'] != 1]  # drop 1 class
    y = df_wine['Class label'].values
    X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values
    from sklearn.model_selection import train_test_split  # 切分训练集与测试集
    from sklearn.preprocessing import LabelEncoder  # 标签化分类变量
    le = LabelEncoder()
    y = le.fit_transform(y)  # 吧y值改为0和1 ，原来是2和3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)  # 2、8分

    ## 我们使用单一决策树分类：
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=None)  # 选择决策树为基本分类器
    from sklearn.metrics import accuracy_score  # 计算准确率
    tree = tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    tree_train = accuracy_score(y_train, y_train_pred)  # 训练集准确率
    tree_test = accuracy_score(y_test, y_test_pred)  # 测试集准确率
    print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))
    # Decision tree train/test accuracies 1.000/0.833

    ## 我们使用BaggingClassifier分类：
    from sklearn.ensemble import BaggingClassifier
    tree = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=None)  # 选择决策树为基本分类器
    bag = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True,
                            bootstrap_features=False, n_jobs=1, random_state=1)
    from sklearn.metrics import accuracy_score
    bag = bag.fit(X_train, y_train)
    y_train_pred = bag.predict(X_train)
    y_test_pred = bag.predict(X_test)
    bag_train = accuracy_score(y_train, y_train_pred)
    bag_test = accuracy_score(y_test, y_test_pred)
    print('Bagging train/test accuracies %.3f/%.3f' % (bag_train, bag_test))
    # Bagging train/test accuracies 1.000/0.917

    '''
    我们可以对比两个准确率，测试准确率较之决策树得到了显著的提高

    我们来对比下这两个分类方法上的差异
    '''
    ## 我们来对比下这两个分类方法上的差异
    x_min = X_train[:, 0].min() - 1
    x_max = X_train[:, 0].max() + 1
    y_min = X_train[:, 1].min() - 1
    y_max = X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    f, axarr = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(12, 6))
    for idx, clf, tt in zip([0, 1], [tree, bag], ['Decision tree', 'Bagging']):
        clf.fit(X_train, y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # ravel()方法将数组维度拉成一维数组,np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等
        Z = Z.reshape(xx.shape)
        axarr[idx].contourf(xx, yy, Z, alpha=0.3)
        axarr[idx].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='blue', marker='^')
        axarr[idx].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='green', marker='o')
        axarr[idx].set_title(tt)
    axarr[0].set_ylabel('Alcohol', fontsize=12)
    plt.tight_layout()
    plt.text(0, -0.2, s='OD280/OD315 of diluted wines', ha='center', va='center', fontsize=12,
             transform=axarr[1].transAxes)
    plt.show()


def wine_regressor():
    from sklearn.datasets import load_boston
    # 从读取房价数据存储在变量 boston 中。
    boston = load_boston()

    # 从sklearn.cross_validation 导入数据分割器。
    from sklearn.model_selection import train_test_split
    X = boston.data
    y = boston.target
    # 随机采样 25% 的数据构建测试样本，其余作为训练样本。
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        random_state=33,
                                                        test_size=0.25)

    # 从 sklearn.preprocessing 导入数据标准化模块。
    from sklearn.preprocessing import StandardScaler
    # 分别初始化对特征和目标值的标准化器。
    ss_X = StandardScaler()
    ss_y = StandardScaler()
    # 分别对训练和测试数据的特征以及目标值进行标准化处理。
    X_train = ss_X.fit_transform(X_train)
    X_test = ss_X.transform(X_test)
    y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
    y_test = ss_y.transform(y_test.reshape(-1, 1))

    # BaggingRegressor
    from sklearn.ensemble import BaggingRegressor
    bagr = BaggingRegressor(n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True,
                            bootstrap_features=False, n_jobs=1, random_state=1)
    bagr.fit(X_train, y_train)
    bagr_y_predict = bagr.predict(X_test)

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    # 使用 R-squared、MSE 以及 MAE 指标对默认配置的随机回归森林在测试集上进行性能评估。
    print('R-squared value of BaggingRegressor:', bagr.score(X_test, y_test))

    print('The mean squared error of BaggingRegressor:', mean_squared_error(
        ss_y.inverse_transform(y_test), ss_y.inverse_transform(bagr_y_predict)))

    print('The mean absoluate error of BaggingRegressor:', mean_absolute_error(
        ss_y.inverse_transform(y_test), ss_y.inverse_transform(bagr_y_predict)))

    '''
    R-squared value of BaggingRegressor: 0.8417369323817341
    The mean squared error of BaggingRegressor: 12.27192314456692
    The mean absoluate error of BaggingRegressor: 2.2523244094488195
    '''

    # 随机森林实现
    from sklearn.ensemble import RandomForestRegressor
    rfr = RandomForestRegressor(random_state=1)

    rfr.fit(X_train, y_train)

    rfr_y_predict = rfr.predict(X_test)

    # 使用 R-squared、MSE 以及 MAE 指标对默认配置的随机回归森林在测试集上进行性能评估。
    print('R-squared value of RandomForestRegressor:', rfr.score(X_test, y_test))

    print('The mean squared error of RandomForestRegressor:', mean_squared_error(
        ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))

    print('The mean absoluate error of RandomForestRegressor:', mean_absolute_error(
        ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))

    '''
    R-squared value of RandomForestRegressor: 0.8083674472512408
    The mean squared error of RandomForestRegressor: 14.859436220472439
    The mean absoluate error of RandomForestRegressor: 2.4732283464566924
    '''

    # 用Bagging集成随机森林
    bagr = BaggingRegressor(base_estimator=rfr, n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True,
                            bootstrap_features=False, n_jobs=1, random_state=1)
    bagr.fit(X_train, y_train)
    bagr_y_predict = bagr.predict(X_test)

    # 使用 R-squared、MSE 以及 MAE 指标对默认配置的随机回归森林在测试集上进行性能评估。
    print('R-squared value of BaggingRegressor:', bagr.score(X_test, y_test))

    print('The mean squared error of BaggingRegressor:', mean_squared_error(
        ss_y.inverse_transform(y_test), ss_y.inverse_transform(bagr_y_predict)))

    print('The mean absoluate error of BaggingRegressor:', mean_absolute_error(
        ss_y.inverse_transform(y_test), ss_y.inverse_transform(bagr_y_predict)))


def multi_plot(names, models, colors, x_train, y_train, x_test, y_test):
    # 绘制整个图像的大小,figure为整体布局大小，dpi表示图片的信息量
    plt.figure(figsize=(20, 20), dpi=200)
    for (name, model, color) in zip(names, models, colors):
        model.fit(x_train, y_train)
        y_test_pred = model.predict(x_test)
        # pos_label表示正样本的值
        fpr, tpr, thresholds = roc_curve(y_test, y_test_pred, pos_label=1)
        accu = accuracy_score(y_test, y_test_pred)
        plt.plot(fpr, tpr, lw=5, label='{}(AUC={:.3f})'.format(name, accu), color=color)
        plt.plot([0, 1], [0, 1], '--', lw=5, color='gray')
        plt.axis('square')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=25)
        plt.ylabel('True Positive Rate', fontsize=25)
        plt.title('ROC Curve', fontsize=30)
        plt.legend(loc='lower right', fontsize=25)
    plt.show()


if __name__ == '__main__':
    test_txt = open('data/horseColicTest.txt').readlines()
    train_txt = open('data/horseColicTraining.txt').readlines()
    # blank_fill(train_txt, test_txt)
    x_train, y_train, x_test, y_test = txt_to_array(train_txt, test_txt)
    tprs = []
    aucs = []
    # 以决策树作为基础树
    svc = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
    DecisionTree = DecisionTreeClassifier(criterion='entropy', random_state=1)
    RandomForest = RandomForestClassifier(random_state=RANDOM_SEED)
    DecisionTreeBagging = BaggingClassifier(base_estimator=DecisionTree, n_estimators=500, max_samples=1.0,
                                            max_features=1.0, bootstrap=True,
                                            bootstrap_features=False, n_jobs=1, random_state=1)
    RandomForestBagging = BaggingClassifier(base_estimator=RandomForest, n_estimators=500, max_samples=1.0,
                                            max_features=1.0, bootstrap=True,
                                            bootstrap_features=False, n_jobs=1, random_state=1)
    models = [svc, DecisionTree, RandomForest, DecisionTreeBagging, RandomForestBagging]
    names = ['svc', 'DecisionTree', 'RandomForest', 'DecisionTreeBagging', 'RandomForestBagging']
    colors = ['red', 'orange', 'green', 'blue', 'pink']
    multi_plot(names, models, colors, x_train, y_train, x_test, y_test)
    # train(RandomForestBagging, x_train, y_train, x_test, y_test)
    # # bagging只适应特征维度为2的数据，所以将所有数据的维度结果pca合成为二维
    # # pca = PCA(n_components=2)
    # # fig = plt.figure(figsize=(10, 8))
    # # x_train = pca.fit_transform(x_train)
    # # gs = gridspec.GridSpec(2, 2)
    # # for clf, tt, grd in zip([DecisionTree, DecisionTreeBagging, RandomForest, RandomForestBagging],
    # #                         ['Decision Tree', 'DecisionTreeBagging', 'Random Forest', 'RandomForestBagging'],
    # #                         itertools.product([0, 1], repeat=2)):
    # #     clf.fit(x_train, y_train)
    # #     ax = plt.subplot(gs[grd[0], grd[1]])
    # #     fig = plot_decision_regions(X=x_train, y=y_train, clf=clf)
    # #     plt.title(tt)
    # # plt.show()
