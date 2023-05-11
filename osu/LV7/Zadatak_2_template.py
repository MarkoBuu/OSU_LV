import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

for i in range(1,7):
    # ucitaj sliku
    img = Image.imread(f"./LV7/imgs/test_{i}.jpg")

    # prikazi originalnu sliku
    plt.figure()
    plt.title("Originalna slika")
    plt.imshow(img)
    plt.tight_layout()
    plt.show()

    # pretvori vrijednosti elemenata slike u raspon 0 do 1 NPR. 158/255=
    img = img.astype(np.float64) / 255                          
    if i == 4:
        img = img[:,:,:3]

    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d)) #Redovi w*h, a stupaca 3 jer je RGB

    # rezultatna slika
    img_array_aprox = img_array.copy()

    #prikazuje nam broj boja
    diff_colors = len(np.unique(img_array_aprox, axis=0))

    print(diff_colors)

    #RANDOM BROJ KLASA
    k = [2,3,4,5,7,10]

    sse = []

    for j in range(len(k)):

        img_array_aprox = img_array.copy()

        km = KMeans( n_clusters = k[j], init ='random', n_init =5 , random_state = 0 )

        km.fit(img_array_aprox)

        #inertia_ zbroj udaljenosti svake tocke od njegovog centra
        sse.append(km.inertia_)

        #npr. imas 3 centra z ide od 0-2
        #km.labels_ sve prediktane vrijednosti da vidimo kojem centru pripadaju
        #SVAKI ELEMENT MIJENJA S VRIJEDNOSTI PRIPADAJUCEG CENTRA
        for z in np.unique(km.labels_):
            #samo true vrijednosti iz array-a se pridruzuju km-cluster_centers_
            img_array_aprox[km.labels_==z] = km.cluster_centers_[z]        

        #Da iz matrice vratis u originalnu velicinu, vraca visinu i duzinu tj. povratak u 3. dimenziju
        img2 = img_array_aprox.reshape(img.shape)

        #3 reda, 2 stupca, index
        plt.subplot(3, 2, j+1)
        plt.title('k = ' + str(k[j]))
        plt.imshow(img2)
    plt.show()


    plt.plot(k, sse)
    plt.show()

    #BINARNO predstavljanje boja, promatramo grupu i izdvojimo ju kao bijelu
    clusters = [3,3,3,3,4,5]

    img_array_aprox = img_array.copy()

    km = KMeans( n_clusters = clusters[i-1], init ='random', n_init =5 , random_state = 0 )

    km.fit(img_array_aprox)

    #fig, axs je mehanizam za stavljanje subplotova u jednom figure-u, mogla sam kao i u61. liniji
    #fig=figure, axs=subplot
    fig, axs = plt.subplots(nrows=1, ncols=clusters[i-1], figsize=(10,5))

    #l je label tj. klasa
    for l, ax in zip(np.unique(km.labels_), axs):
        #DA MI SLIKA BUDE CIJELA CRNA
        layer_arr = np.zeros_like(km.labels_)
        #mjenjam samo odabranu l klasu u BIJELU tj. 1
        layer_arr[km.labels_ == l] = 1
        #vracanje u pocetne velicine bez dubine
        layer_img = layer_arr.reshape(w, h)
        #za prikaz subplot-a, gray kada ocemo da bude crno-bijelo
        ax.imshow(layer_img, cmap='gray')
    plt.suptitle('the primary group is represented in white')
    plt.show()
