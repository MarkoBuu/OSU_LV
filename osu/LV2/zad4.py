import numpy as np
import matplotlib.pyplot as plt 

zeros = np.zeros((50,50))#CRNA slika
ones = np.ones((50,50))#BIJELA slika
lm = np.vstack((zeros,ones))#na stack po stupcu prvo crne pa bijele
rm = np.vstack((ones, zeros))#na stack bijele pa crne
m = np.hstack((lm,rm))#hstack koji ide po retku stavio lm i rm jedno do druge


plt.figure()
plt.imshow(m, cmap='gray')
plt.show()
