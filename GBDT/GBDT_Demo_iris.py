import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.tree import export_text

import filter.param

iris = load_iris()  # 加载鸢尾花数据集
data = iris.data  # 特征数据
target = iris.target  # 分类数据

GBDTreg = GradientBoostingClassifier(**filter.param.params)
GBDTreg.fit(data, target)
for ii in range(0, GBDTreg.n_estimators):
    for jj in range(0, 3):
        sub_tree = GBDTreg.estimators_[ii, jj]  # GBDTreg.estimators_.shape = (2,3)
        plt.figure(figsize=(15, 9))
        plot_tree(sub_tree)
        r1 = export_graphviz(sub_tree)
        print(r1)
        r2 = export_text(sub_tree, feature_names=iris['feature_names'])
        print(r2)
y_predict = GBDTreg.predict(data)
