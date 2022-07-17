# 排列特征重要性# 模型选用不支持特征选择的
from matplotlib import pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


# 定义函数
def plot_importance(model, X, y, scoring):
    model.fit(X, y)
    results = permutation_importance(model, X, y, scoring=scoring)
    importance = results.importances_mean
    if importance.ndim == 2: importance = importance[0]
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
        plt.bar([*range(len(importance))], importance)
        plt.show()


