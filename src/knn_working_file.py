'''
   Neste tutorial sobre aprendizado de máquina, apresentarei o algoritmo K-Nearest 
   Neighbors. Este é um algoritmo de classificação que tenta classificar os pontos 
   de dados com base em seus 'vizinhos' mais próximos.

   Distância euclidiana e como os pontos são classificados, este algoritimo tem prós 
   e contras.
'''

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
import numpy as np 
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

le = preprocessing.LabelEncoder()

# buying,maint,door,persons,lug_boot,safety,class
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
classe = le.fit_transform(list(data["class"]))

#print(buying)
predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
Y = list(classe)

x_train, x_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=9) 
# 5 resulta em: 0.8670520231213873
# 7 resulta em: 0.953757225433526
# 7 resulta em: 0.9190751445086706
# 7 resulta em: 0.9826589595375722

model.fit(x_train, Y_train)
acc = model.score(x_test, Y_test)
print(acc)


predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    #print("Predicted: ", predicted[x], " Data: ", x_test[x], " Actual: ", Y_test[x])
    print("Predicted: ", names[predicted[x]], " Data: ", x_test[x], " Actual: ", names[Y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)
    