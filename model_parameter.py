import numpy as np
from mlxtend.classifier import StackingCVClassifier
from numpy import ndarray
from pyparsing import Dict
from sklearn import model_selection, svm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, StackingClassifier, BaggingClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from traitlets import List
from univariate_logistic_regression import Node, RegressionTree

from bagging.stacking import RANDOM_SEED

fold5_GBDT = GradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.18,
    n_estimators=100,
    max_features=0.8,
    max_depth=3,
    verbose=2
)

# 0.76
GBR = GradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.15,
    n_estimators=500,
    max_features=None,
    max_depth=5,
    verbose=2
)

# 0.66
GBC = GradientBoostingClassifier(loss="log_loss",
                                 learning_rate=0.1,
                                 n_estimators=600,
                                 subsample=0.8,
                                 criterion="friedman_mse",
                                 max_depth=4)

# auc:0.6082865427098674
# ks:0.2165730854197349
clf1 = KNeighborsClassifier(n_neighbors=3)
# auc:0.5399317755415317
# ks:0.07986355108306328
clf2 = RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=150, max_depth=None, max_features="auto")
# auc:0.5763228531120028
# ks:0.15264570622400564
clf3 = GaussianNB()
# auc:0.5812087342010942
# ks:0.16241746840218826
lr = LogisticRegression()
# Starting from v0.16.0, StackingCVRegressor supports
# `random_state` to get deterministic result.
# auc:0.5511465138757876
# ks:0.10229302775157507
stacking = StackingCVClassifier(classifiers=[clf1, clf2, clf3],  # 第一层分类器
                                meta_classifier=lr,
                                # 第二层分类器，并非表示第二次stacking，而是通过logistic regression对新的训练特征数据进行训练，得到predicted label
                                random_state=RANDOM_SEED)

# auc:0.6147184336038826
# ks:0.22943686720776502
tree = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=None)  # 选择决策树为基本分类器
# auc:0.5776287452299089
# ks:0.15525749045981788
bagging = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples=0.8, max_features=1.0, bootstrap=True,
                            bootstrap_features=False, n_jobs=1, random_state=1)

# auc:0.5562078008198699
# ks:0.11241560163973957
training_RFC = RandomForestClassifier()
param_grid = [
    {'n_estimators': [100, 150, 200, 300, 400, 450, 500, 550, 600, 650, 700, 800, 1000], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
grid_search = GridSearchCV(training_RFC, param_grid, cv=5, scoring='roc_auc')

# auc:0.7831300903632363
# ks:0.4302731704417098
training_RFR = GradientBoostingRegressor()

parameters = {'C': [0.05, 0.1, 0.5, 1, 2, 5]}
base_estimators = SVC()
# 报错：Liblinear failed to converge, increase the number of iterations.
# 解决办法1：增加max_iter,LinearSVC(max_iter=5000)
# 解决办法2：取消默认值，dual=False,LinearSVC(dual=False)
# auc:0.601288641393589
# ks:0.2025772827871779
grid_linear_svc = model_selection.GridSearchCV(estimator=LinearSVC(dual=False), param_grid=parameters,
                                               scoring='accuracy',
                                               cv=5, verbose=1)

GBDT_ = GradientBoostingClassifier(loss='log_loss', learning_rate=0.1, n_estimators=5, subsample=1
                                   , min_samples_split=2, min_samples_leaf=1, max_depth=3
                                   , init=None, random_state=None, max_features=None
                                   , verbose=0, max_leaf_nodes=None, warm_start=False
                                   )
