#import bibilioteka
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay


def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x3_min, x3_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x4_min, x4_max = X[:, 1].min() - 1, X[:, 1].max() + 1
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


##################################################
#1. zadatak
##################################################

#učitavanje dataseta
data = pd.read_csv('titanic.csv')
#print(data)
#a)
print("Number of passengers: ", len(data))
#b)
print("Number of survived: ", len(data[data['Survived'] == 1]))
#c)

plt.figure()
grouped = data.groupby('Sex')
grouped['Survived'].count().plot(kind='bar')
plt.title("Number of survived by sex")
plt.ylabel("sex")
plt.xlabel("survived")
plt.show()


#d)
m = data[data['Sex'] == "male"]
print("Average age male: ", (m['Age'].sum()/len(m)))
f = data[data['Sex'] == "female"]
print("Average age female: ", (f['Age'].sum()/len(f)))
#e)
m = data[data['Sex'] == "male"]
m = m[m['Survived'] == 1]
print("Average age survived male: ", (m['Age'].sum()/len(m)))
f = data[data['Sex'] == "female"]
f = f[f['Survived'] == 1]
print("Average age survived female: ", (f['Age'].sum()/len(f)))
#dodati da su prezivjeli
m1 = m[m['Pclass']==1]
m1 = m1[m1['Sex']== "male"]
print("Youngest male in Pclass 1: ", (m1['Age'].min()))
m2 = m[m['Pclass']==2]
m2 = m2[m2['Sex']=="male"]
print("Youngest male in Pclass 2: ", (m2['Age'].min()))
m3 = m[m['Pclass']==3]
m3 = m3[m3['Sex']== "male"]
print("Youngest male in Pclass 3: ", (m3['Age'].min()))

print('-'*50)
##################################################
#2. zadatak
##################################################

#učitavanje dataseta
data = pd.read_csv('titanic.csv')
data = pd.get_dummies(data, columns=['Sex'])
data = data.drop(["Sex_male"], axis=1)
data= data.rename(columns={"Sex_female": "Sex"})
data = data[['Pclass', 'Sex', 'Fare', 'Embarked', 'Survived']]
data['Embarked'].replace({'S' : 0,
                        'Q' : 1,
                        'C': 2}, inplace = True)
print(data)
data = data.dropna(axis = 0)
data = data.reset_index(drop = True)
X = data[['Pclass', 'Sex', 'Fare', 'Embarked']]
y = data['Survived']
#embarked q,s,c
#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)

sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

#a)
KNN_model = KNeighborsClassifier(n_neighbors = 5)
KNN_model.fit(X_train_n , y_train)
#plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.figure()
plt.title("Dead depending on sex and fare")
plt.xlabel("Fare")
plt.ylabel("Sex")
plt.scatter(X_train['Fare'], X_train['Sex'], c=y_train)
plt.show()
#b)
y_train_p = KNN_model.predict(X_train_n)
y_test_p = KNN_model.predict(X_test_n)
print("KNN k=5: ")
print("Točnost train: ", accuracy_score(y_train, y_train_p))
print("Točnost test: ", accuracy_score(y_test, y_test_p))
#c)
print("KNN unakrsna validacija: ")
KNN_model2 = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 100)}
KNN_gs = GridSearchCV(KNN_model2, param_grid, cv=5)
KNN_gs.fit(X_train_n, y_train)
print(KNN_gs.best_params_, KNN_gs.best_score_)
#d)
y_train_p = KNN_gs.predict(X_train_n)
y_test_p = KNN_gs.predict(X_test_n)
print("Točnost nakon unakrsne validacije: ")
print("Točnost train: ", accuracy_score(y_train, y_train_p))
print("Točnost test: ", accuracy_score(y_test, y_test_p))

print('-'*50)
##################################################
#3. zadatak
##################################################

#učitavanje podataka:
data = pd.read_csv('titanic.csv')
data = pd.get_dummies(data, columns=['Sex'])
data = data.drop(["Sex_male"], axis=1)
data= data.rename(columns={"Sex_female": "Sex"})
data = data[['Pclass', 'Sex', 'Fare', 'Embarked', 'Survived']]
data['Embarked'].replace({'S' : 0,
                        'Q' : 1,
                        'C': 2}, inplace = True)
print(data)
data = data.dropna(axis = 0)
data = data.reset_index(drop = True)
X = data[['Pclass', 'Sex', 'Fare', 'Embarked']]
y = data['Survived']

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

#a)
model = keras.Sequential()
model.add(layers.Input(shape = (4, )))
model.add(layers.Dense(16, activation = "relu"))
model.add(layers.Dense(8, activation = "relu"))
model.add(layers.Dense(4, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()

#b)
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy", ])
#c)
batch_size = 5
epochs = 100
history = model.fit(X_train_n , y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.1)
predictions = model.predict(X_test)
score = model.evaluate(X_test, y_test ,verbose = 0)
print(score)
#d)
model.save("FCNispit1/")
del model
#e)
model = load_model("FCNispit1/")
model.summary ()

score = model.evaluate(X_test, y_test, verbose = 0)
print("Evaluate model: ", score)
#f)
predictions = model.predict(X_test)
predictions = np.round(predictions)
cm = confusion_matrix(y_test , predictions)
print ("Confusion matrix : " , cm )
disp = ConfusionMatrixDisplay(confusion_matrix(y_test , predictions))
disp.plot()
plt.show()
