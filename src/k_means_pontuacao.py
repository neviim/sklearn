'''
    Pontuação:
    
    Para pontuar nosso modelo, usaremos uma função no site sklearn. Ele calcula muitas pontuações 
    diferentes para diferentes partes do nosso modelo. Se você quiser saber mais sobre o significado 
    desses valores, visite o seguinte site.
'''

import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
data = scale(digits.data)
y = digits.target

k = 10
samples, features = data.shape


def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

'''
Treinando o modelo:

Finalmente, para treinar o modelo, criaremos um classificador K Means e, 
em seguida, passamos esse classificador para a função que criamos acima 
para pontuá-lo e treiná-lo. 
(https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation)
'''
clf = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(clf, "1", data)