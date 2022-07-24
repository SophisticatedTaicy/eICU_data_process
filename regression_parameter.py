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

from bagging.stacking import RANDOM_SEED

GBDTreg_vali = GradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.18,
    n_estimators=100,
    max_features=0.8,
    max_depth=3,
    verbose=2
)

params = {'n_estimators': 500,  # 弱分类器的个数
          'max_depth': 3,  # 弱分类器（CART回归树）的最大深度
          'min_samples_split': 5,  # 分裂内部节点所需的最小样本数
          'learning_rate': 0.05,  # 学习率
          'loss': 'squared_error'}  # 损失函数：均方误差损失函数
GBDTreg = GradientBoostingRegressor(**params)

# 0.76
GBDTreg = GradientBoostingRegressor(
    loss="squared_error",
    learning_rate=0.15,
    n_estimators=500,
    max_features=None,
    max_depth=5,
    verbose=2
)

parameters = {'C': [*np.linspace(0.1, 1, 5)], 'solver': ('liblinear', 'saga', 'newton-cg', 'lbfgs', 'sag')
              #               ,'max_iter':[100,150,200]
              }
LR = LogisticRegression(max_iter=1000, penalty='l2', C=0.1, solver='liblinear')
LR = GridSearchCV(LR, parameters, cv=10)
