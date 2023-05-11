import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Učitavanje Iris dataset
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Pretvaranje ciljne varijable u kategoričke vrijednosti
encoder = LabelEncoder()
encoded_y = encoder.fit_transform(y)
dummy_y = to_categorical(encoded_y)

# Podijela podataka na skup za učenje i skup za testiranje
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size=0.25, random_state=42)

# a) Izgradnja neuronske mreže
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(5, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))

# Ispis informacija o mreži
model.summary()

# b) Podešavanje procesa treniranja
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# c) Učenje mreže
epochs = 500
batch_size = 7
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# d) Pohranjivanje modela na tvrdi disk
model.save('iris_model.h5')

# e) Evaluacija mreže na testnom skupu podataka
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Točnost evaluacije na testnom skupu: {:.2f}%".format(accuracy * 100))

# f) Predikcija mreže na skupu podataka za testiranje i prikaz matrice zabune
from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

confusion_mtx = confusion_matrix(y_test_classes, y_pred_classes)

print("Matrica zabune:")
print(confusion_mtx)
