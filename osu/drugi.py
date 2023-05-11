import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets

# Učitavanje Iris dataset
iris = datasets.load_iris()
X = iris.data
true_labels = iris.target

# a) Pronalaženje optimalnog broja klastera K
k_values = range(1, 11)
inertia_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

# b) Graficki prikaz lakat metode
plt.plot(k_values, inertia_values, marker='o')
plt.title('Lakat metoda')
plt.xlabel('Broj klastera (K)')
plt.ylabel('Inercija')
plt.show()

# c) Primjena algoritma K srednjih vrijednosti s optimalnim brojem klastera K
optimal_k = 3  # Zamijenite s odabranom vrijednosti za K
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
# inicijalizacija algoritma K srednjih vrijednosti


# d) Prikaz dobivenih klastera pomoću dijagrama raspršenja
colors = ['red', 'blue', 'green']  # Zamijenite s odgovarajućim bojama za vaš broj klastera

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.title('K-means klasteri')
plt.xlabel('Duljina latica')
plt.ylabel('Širina čašice')

# Dodavanje legende
legend_labels = [f'Klaster {i+1}' for i in range(optimal_k)]
plt.legend(legend_labels, loc='upper right')

plt.show()

# e) Usporedba dobivenih klasa s njihovim stvarnim vrijednostima i izračun točnosti klasifikacije
accuracy = np.mean(labels == true_labels) * 100
print("Točnost klasifikacije: {:.2f}%".format(accuracy))
