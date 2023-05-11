import numpy as np
import matplotlib.pyplot as plt 

#posvijetljena slika
img = plt.imread ("./LV2/road.jpg")
plt.figure()
plt.imshow(img, alpha=0.8)#alpha 0.5 je za posvijetljenje, sto je alpha manji slika je SVJETLIJA

#slika je rotirana u desno za 90°
imgr = np.rot90(img, axes=(0,1))#rotirana pomoću axes, parametri na axes nam govore u koju stranu, ako je 1 na desno ide u lijevo, ako je 1 na lijevo ide u desno
plt.figure()
plt.imshow(imgr)

#zrcali sliku po y osi
imgf = np.flip(img, axis=1)#za to zadužena funkcija flip, kada je 1 rotira se prema y, a 0 po x
plt.figure()
plt.imshow(imgf)

#ostavili smo 2. četvrtinu slike
rows, cols, pixels = img.shape
imgc = img[:,round(cols/4):round(cols/2),:].copy()#od 1. četvrtine do 2. četvrtine od ukupnog broja kolona
plt.figure()
plt.imshow(imgc)
plt.show()
