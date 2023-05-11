import numpy as np
import matplotlib.pyplot as plt


x = np.array([1, 2, 3, 3, 1], float)
y = np.array([1, 2, 2, 1, 1], float)
plt.plot(x, y, 'r', linewidth=1, marker=".", markersize=10)
plt.axis([0, 4, 0, 4])
plt.xlabel('x os')
plt.ylabel('y os')
plt.title('ZADATAK 1')
plt.show()