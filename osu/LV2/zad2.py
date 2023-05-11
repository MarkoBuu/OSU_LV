import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt('./LV2/data.csv', skiprows=1, delimiter=',')
print(data)

#a)koliko redaka imam toliko ljudi imam
print('The amount of people:')
print(data.shape[0])

#b)
h = data[:, 1] #podatke iz stupca cijelog 1.
w = data[:, 2] #podatke iz stupca cijelog 1.
plt.xlabel('height')
plt.ylabel('weight')
plt.title('ZADATAK_1_b')
plt.scatter(h, w, color = 'hotpink')
plt.show()

#c)
h=data[:51, 1] #mjerenje za svaku pedesetu osobu
w=data[:51, 2]
plt.xlabel('height')
plt.ylabel('weight')
plt.title('ZADATAK_1_c')
plt.scatter(h, w, color = 'hotpink')
plt.show()

m = (data[:, 0] == 1)
z = (data[:, 0] == 0)

print('man min hight', data[m,1].min())
print('man max hight', data[m,1].max())
print('man mean hight', data[m,1].mean())

print('woman min hight', data[z,1].min())
print('woman max hight', data[z,1].max())
print('woman mean hight', data[z,1].mean())

