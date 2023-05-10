import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,     #BINARNA KLASIFIKACIJA
                            random_state=213, n_clusters_per_class=1, class_sep=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

plt.figure()
plt.scatter(x=X_train[:, 0], y=X_train[:, 1], c=y_train, cmap="coolwarm")
plt.scatter(x=X_test[:, 0], y=X_test[:, 1], c=y_test, cmap="coolwarm", marker="x")
plt.show()


#b)
LogRegression_model = LogisticRegression()                                                                  #MODEL LOGISTIČKE REGRESIJE
LogRegression_model.fit(X_train, y_train)

print("Koeficjent", LogRegression_model.coef_)
print("Prosječna točka", LogRegression_model.intercept_)                                                    #THETA0


#c)
theta0 = LogRegression_model.intercept_                                                                     #THETA0
theta1 = LogRegression_model.coef_[0,0]                                                                     #THETA1
theta2 = LogRegression_model.coef_[0,1]                                                                     #THETA2

x_min, x_max = np.min(X_train[:, 1]), np.max(X_train[:, 1])

x2 = np.linspace(x_min, x_max, 100)                                                                         #X2 nam je na y
x1 = -theta0/theta1 -theta2/theta1*x2                                                                       #obrnem formulu da nađem X1

plt.plot(x1, x2)

plt.fill_between(x1, x2, x_min, alpha=0.2, color='blue')                                                    #DA OBOJI S JEDNE STRANE JEDNU KLASU S DRUGE DRUGU
plt.fill_between(x1, x2, x_max, alpha=0.2, color='red')

plt.show()


#d)
y_pred = LogRegression_model.predict(X_test)                                                                #MORAMO PREDICTAT TEST SKUP DA VIDIMO KAK RADI
cm=confusion_matrix( y_test, y_pred )                                                                       #MATRICA ZABUNE
print(" Matrica zabune : ", cm )
disp = ConfusionMatrixDisplay( confusion_matrix(y_test , y_pred ) )
disp.plot()
plt.show ()

#e)
correctly_classified = y_test ==  y_pred
plt.scatter(
    X_test[correctly_classified, 0],                                                                        #LIJEVO redovi DESNO stupci, uzela 1. supac
    X_test[correctly_classified, 1],                                                                        #uzela 2. stupac i prikatala ih kao koordinate u scatterplot-u
    c = "green",
    marker = "o"
)

incorrectly_classified = y_test != y_pred
plt.scatter(
    X_test[incorrectly_classified, 0],
    X_test[incorrectly_classified, 1],
    c="black",
    marker="x"
)

plt.show()
