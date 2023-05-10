import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X



k=[3, 3, 4, 2, 2]

for i in range(1, 6):
    #generiramo podatke
    x=generate_data(500, i)
    #inicializacija algoritma K srednjih vrijednosti
    #koliko imamo centara, nacin inicializacije centara(default k-means++), koliko puta ce se izvrsiti algoritam i random state
    km = KMeans(n_clusters=k[i-1], init ='random', n_init =5 , random_state =0)
    km.fit(x)
    labels = km.predict(x)

    # prikazi primjere u obliku dijagrama rasprsenja
    plt.figure()
    #na koliko podjela ide toliko boja imamo
    #prvi stupac, drugi stupac i treća kolona je boja
    #koliko prediktanih centara ima imati ce toliko razlicitih boja
    plt.scatter(x[:,0], x[:,1], c=labels)                        
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('podatkovni primjeri')
    plt.show()