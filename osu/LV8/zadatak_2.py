import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.models import load_model
import random

num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255


# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)


# Reshaped arrays from (60000, 28, 28, 1) to (60000, 784)
x_train_s=x_train_s.reshape(x_train_s.shape[0],-1)
x_test_s=x_test_s.reshape(x_test_s.shape[0],-1)


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


# Ucitavanje modela
model= load_model('ImageReader/')
predictions=model.predict(x_test_s)


# Pretvaranje podataka nazad u brojeve
predictions=(predictions >=0.5).astype(int)
#pretvaramo nazad u NUMERIČKE
y_test_s=y_test_s.astype(int)
#axis=1 gleda kolone, argmax dohvaca vrijednost najveceg indexa u polju i prema tome znamo pod koju klasu spada
y_pred=np.argmax(predictions, axis=1)
y_true=np.argmax(y_test_s, axis=1)


# Pronalazenje lose klasificiranih podataka
miss_index_list=[]
#shape[0]= 60 000
for i in range(x_test_s.shape[0]):
    if y_pred[i]!=y_true[i]:
        miss_index_list.append(i)


# Crtanje lose klasificiranih podataka
for _ in range(3):
    plt.figure()
    i=random.choice(miss_index_list)
    #dohvatio red i dohvacas zapravo pod tim jednu vrijednost od 0-60 000 jer je toliko podataka
    plt.imshow(x_test[i,:,:])
    plt.title(f'Predicted number: {y_pred[i]}       Real number: {y_true[i]}')
    plt.show()

