import numpy as np
from mlxtend.classifier import StackingCVClassifier
from sklearn import model_selection, svm
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import scipy.stats as stats
from sklearn.utils.fixes import loguniform
from xgboost import XGBClassifier
from skopt import BayesSearchCV
# 参数范围由下面的一个指定
from skopt.space import Real, Categorical, Integer

from bagging.stacking import RANDOM_SEED

# 0.66
GBDTclas = GradientBoostingClassifier(loss="log_loss",
                                      learning_rate=0.1,
                                      n_estimators=600,
                                      subsample=0.8,
                                      criterion="friedman_mse",
                                      max_depth=4)

# auc:0.6082865427098674
# ks:0.2165730854197349
k_value = [55, 57, 59, 60, 62, 65]
knn = KNeighborsClassifier()
grid_param = {'n_neighbors': list(range(50, 65))
              }
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
KNNclas = KNeighborsClassifier()
KNNclas_GridSearchCV = GridSearchCV(KNNclas, grid_param, cv=5)
KNNclas_RandomizedSearchCV = RandomizedSearchCV(KNNclas, grid_param, n_iter=13)
# auc:0.5399317755415317
# ks:0.07986355108306328
RFclas = RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=150, max_depth=None, max_features="sqrt")
# auc:0.5763228531120028
# ks:0.15264570622400564
GNB = GaussianNB()
# 还可以使用网格搜索调参
# 网格搜索进行调参
parameters = {'C': [*np.linspace(0.1, 1, 5)], 'solver': ('liblinear', 'saga', 'newton-cg', 'lbfgs', 'sag')
              #               ,'max_iter':[100,150,200]
              }
LR = LogisticRegression(max_iter=1000, penalty='l2', C=0.1, solver='liblinear')
LR = GridSearchCV(LR, parameters, cv=10)
from sklearn.model_selection import GridSearchCV

# auc:0.5511465138757876
# ks:0.10229302775157507
stackingclas = StackingCVClassifier(classifiers=[KNNclas, RFclas, GNB],  # 第一层分类器
                                    meta_classifier=LR,
                                    # 第二层分类器，并非表示第二次stacking，而是通过logistic regression对新的训练特征数据进行训练，得到predicted label
                                    random_state=RANDOM_SEED)

# auc:0.6147184336038826
# ks:0.22943686720776502
DTclas = DecisionTreeClassifier(
    criterion="log_loss",
    splitter="best",
    max_depth=3,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=321432)  # 选择决策树为基本分类器
# auc:0.5776287452299089
# ks:0.15525749045981788
baggingclas = BaggingClassifier(base_estimator=DTclas, n_estimators=500, max_samples=0.8, bootstrap=True,
                                bootstrap_features=False, n_jobs=1, random_state=123)

# auc:0.5562078008198699
# ks:0.11241560163973957
RFC = RandomForestClassifier()
param_grid = [
    {'n_estimators': [100, 150, 200, 300, 400, 450, 500, 550, 600, 650, 700, 800, 1000], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
RFclas = GridSearchCV(RFC, param_grid, cv=5, scoring='roc_auc')

# 报错：Liblinear failed to converge, increase the number of iterations.
# 解决办法1：增加max_iter,LinearSVC(max_iter=5000)
# 解决办法2：取消默认值，dual=False,LinearSVC(dual=False)
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
}
param_dist = {
    'C': stats.uniform(0.1, 1e4),
    'gamma': loguniform(1e-6, 1e+1),
}
SVC_GridSearchCV = model_selection.GridSearchCV(SVC(), param_grid=param_grid,
                                                refit=True,
                                                scoring='accuracy',
                                                cv=5, verbose=1)
SVC_RandomizedSearchCV = RandomizedSearchCV(
    SVC(),
    param_distributions=param_dist,
    n_iter=20,
    refit=True,
    verbose=3)
search_spaces = {
  'C': Real(0.1, 1e+4),
  'gamma': Real(1e-6, 1e+1, 'log-uniform'),
}
# 0.75
GBDTclas = GradientBoostingClassifier(loss='log_loss', criterion='friedman_mse', max_features='log2')
GBDT_params = {
    'max_depth': [3, 5, 7],
    'n_estimators': [1000, 2000, 3000],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.15]
}
GBDT_GridSearchCV = GridSearchCV(GBDTclas, param_grid=GBDT_params, scoring='accuracy', cv=5)
GBDT_RandomizedSearchCV = RandomizedSearchCV(GBDTclas, GBDT_params, n_iter=10)

# 调参方式和逻辑回归一样，可以使用学习曲线和网格搜索。

XGBclas = XGBClassifier()
# 4.2 使用5折交叉验证评估模型参数
XGB_params = {
    'max_depth': [3, 5, 6, 7],
    'n_estimators': [200, 500, 700, 1000],
    'learning_rate': [0.01, 0.1, 0.15],
    'min_child_weight': [1, 2, 3]
}
XGBclas_GridSearch = GridSearchCV(XGBclas, param_grid=XGB_params, scoring='accuracy', cv=5)
