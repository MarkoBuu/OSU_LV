import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# Učitavanje Iris dataset
iris = datasets.load_iris()

# a) Prikaži odnos duljine latice i čašice za klase Versicolour i Virginica
versicolor_data = iris.data[iris.target == 1]
versicolor_labels = iris.target[iris.target == 1]
virginica_data = iris.data[iris.target == 2]
virginica_labels = iris.target[iris.target == 2]

plt.scatter(versicolor_data[:, 2], versicolor_data[:, 3], c='blue', label='Versicolour')
plt.scatter(virginica_data[:, 2], virginica_data[:, 3], c='red', label='Virginica')

plt.title('Odnos duljine latice i čašice')
plt.xlabel('Duljina latice')
plt.ylabel('Širina čašice')

plt.legend()

plt.show()

# b) Prikaz prosječne vrijednosti širine čašice za sve tri klase
setosa_data = iris.data[iris.target == 0]
versicolor_data = iris.data[iris.target == 1]
virginica_data = iris.data[iris.target == 2]

setosa_mean = setosa_data[:, 1].mean()
versicolor_mean = versicolor_data[:, 1].mean()
virginica_mean = virginica_data[:, 1].mean()

labels = ['Setosa', 'Versicolour', 'Virginica']
means = [setosa_mean, versicolor_mean, virginica_mean]

plt.bar(labels, means)

plt.title('Prosječna širina čašice za svaku klasu')
plt.xlabel('Klase cvijeta')
plt.ylabel('Prosječna širina čašice')

plt.show()

# c) Broj jedinki pripadnika klase Virginica s većom širinom čašice od prosjeka
virginica_widths = iris.data[iris.target == 2][:, 3]
virginica_mean = np.mean(virginica_widths)
count = np.sum(virginica_widths > virginica_mean)

print("Broj jedinki pripadnika klase Virginica s većom širinom čašice od prosjeka:", count)
