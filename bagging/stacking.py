from matplotlib import pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools

RANDOM_SEED = 42

if __name__ =='__main__':
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=RANDOM_SEED)
    clf3 = GaussianNB()
    lr = LogisticRegression()

    # Starting from v0.16.0, StackingCVRegressor supports
    # `random_state` to get deterministic result.
    sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],  # 第一层分类器
                                meta_classifier=lr,
                                # 第二层分类器，并非表示第二次stacking，而是通过logistic regression对新的训练特征数据进行训练，得到predicted label
                                random_state=RANDOM_SEED)

    params = {'kneighborsclassifier__n_neighbors': [1, 5],
              'randomforestclassifier__n_estimators': [10, 50],
              'meta_classifier__C': [0.1, 10.0]}

    # 堆叠5折CV分类与网格搜索(结合网格搜索调参优化)
    grid = GridSearchCV(estimator=sclf,
                        param_grid=params,
                        cv=5,
                        refit=True)

    print('3-fold cross validation:\n')

    for clf, label in zip([clf1, clf2, clf3, sclf], ['KNN', 'Random Forest', 'Naive Bayes', 'StackingClassifier']):
        scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))



    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(10, 8))
    for clf, lab, grd in zip([clf1, clf2, clf3, sclf],
                             ['KNN',
                              'Random Forest',
                              'Naive Bayes',
                              'StackingCVClassifier'],
                             itertools.product([0, 1], repeat=2)):
        clf.fit(X, y)
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X=X, y=y, clf=clf)
        plt.title(lab)
    plt.show()