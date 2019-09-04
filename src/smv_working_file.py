'''
    Neste tutorial em python de aprendizado de máquina, apresentarei as Máquinas de 
    vetores de suporte. Isso é usado principalmente para classificação e é capaz de 
    executar a classificação para dados dimensionais grandes. Também mostrarei como 
    carregar conjuntos de dados diretamente do módulo sklearn.

    Neste tutorial em python do aprendizado de máquina, explique como funciona uma 
    máquina de vetores de suporte. O SVM funciona criando um hiperplano que divide 
    os dados de teste em suas classes. Então, olho para qual lado do hiperplano está 
    um ponto de dados de teste e o classifico.

    Doc:
        sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
'''

import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

#print(x_train, y_train)
classes = ["malignant", "benign"]

#clf = svm.SVC()                            # com este parametro acc = 0.6578947368421053
#clf = svm.SVC(kernel="linear")             # com este parametro acc = 0.9736842105263158
#clf = svm.SVC(kernel="linear", C=1)        # com este parametro acc = 0.956140350877193
#clf = svm.SVC(kernel="linear", C=2)        # com este parametro acc = 0.9298245614035088
#clf = KNeighborsClassifier(n_neighbors=9)  # com este parametro acc = 0.9473684210526315
clf = KNeighborsClassifier(n_neighbors=13)  # com este parametro acc = 0.9210526315789473
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)

