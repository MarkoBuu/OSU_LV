import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score


labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

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
                    edgecolor = 'w',
                    label=labels[cl])
    plt.show()

# ucitaj podatke
df = pd.read_csv("LV5/penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()[:,0]

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)


#a)
unique_train, count_train = np.unique(y_train, return_counts=True)                                  #vraca polje koje postoje sve vrste 0,1 i 2, count napise koliko ima unikatnih za svaku vrijednost
unique_test, count_test = np.unique(y_test, return_counts=True)

# plot bar chart
plt.bar(unique_train, count_train, alpha=0.5, color='green', label='Podaci za učenje')              #sto je alfa veća
plt.bar(unique_test, count_test, alpha=0.5, color='magenta', label='Test podaci')
plt.xticks(unique_train)                                                                            #label za svaki stick/stupac
plt.xlabel('Vrsta')
plt.ylabel('Broj podataka')
plt.legend()

plt.show()


#b)
LogRegression_model = LogisticRegression()
LogRegression_model.fit( X_train , y_train )


coef = LogRegression_model.coef_                                                                    #PARAMETRI MODELA
intercept = LogRegression_model.intercept_                                                          #NULTA THETA PARAMETAR MODELA

print(coef)
print(intercept)

#d)
plot_decision_regions(X_train, y_train, LogRegression_model)

#e)
y_pred = LogRegression_model.predict(X_test)                                                        #PROVEDITE KLASIFIKACIJU = PREDIKCIJA TESTNOG SKUPA

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
disp.plot()
plt.show()

print(f'Točnost: {accuracy_score(y_test,y_pred)}')                                                  #accuracy_score računa nam TOČNOST
print(classification_report(y_test,y_pred))                                                         #izračunava 4 glavne metrike (ovdje zadano da izračunamo na testnim)
#               precision    recall  f1-score   support

#            0       0.96      0.89      0.92        27
#            1       0.94      0.88      0.91        17
#            2       0.89      1.00      0.94        25

#     accuracy                           0.93        69
#    macro avg       0.93      0.92      0.93        69
# weighted avg       0.93      0.93      0.93        69


#f)
# dodavanje parametara
input_variables = ['bill_length_mm',
                    'flipper_length_mm',
                    'bill_depth_mm',
                    'body_mass_g']

X = df[input_variables].to_numpy()
#ipsilon uvijek samo 1 kolona jer je izlaz
y = df[output_variable].to_numpy()[:,0]                                                             #uzela je samo 1. kolonu

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

y_pred = LogRegression_model.predict(X_test)                                                        

print(classification_report(y_test, y_pred))
#               precision    recall  f1-score   support

#            0       1.00      0.93      0.96        27
#            1       0.89      1.00      0.94        17
#            2       1.00      1.00      1.00        25

#     accuracy                           0.97        69
#    macro avg       0.96      0.98      0.97        69
# weighted avg       0.97      0.97      0.97        69




