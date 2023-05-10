import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("./LV6/Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))



#3 ZADATAK
# inicijalizacija i ucenje SVM modela
SVM_model=svm.SVC(kernel='rbf', gamma=1, C=0.1)
SVM_model.fit( X_train_n, y_train )

# predikcija na skupu podataka za testiranje
y_test_p_SVM=SVM_model.predict( X_test )
y_train_p_SVM=SVM_model.predict( X_train )

# granica odluke pomocu KNN regresije
plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
plt.tight_layout()
plt.show()

#param_grid = {'C': [1, 10, 100 ],
#              'gamma': [10, 1, 0.1, 0.01],}
param_grid = {'C': [1],
              'gamma': [1],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
#optimalni parametar za SVM: c, gamma i kernel
SVM_gs = GridSearchCV(SVM_model, param_grid, cv=5)
SVM_gs.fit(X_train_n, y_train)
print(SVM_gs.best_params_, SVM_gs.best_score_)

C=[1, 10, 100 ]
gamma = [10, 1, 0.1, 0.01 ]

for i in range(len(C)):
    for j in range(len(gamma)):
        SVM_model = svm.SVC(kernel = 'rbf', gamma=gamma[j], C=C[i])
        SVM_model.fit(X_train_n, y_train)
        plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
    plt.show()



