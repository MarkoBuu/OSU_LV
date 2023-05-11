import numpy as np
from tensorflow import keras
from PIL import Image
from matplotlib import pyplot as plt

model = keras.models.load_model('./LV8/FCN/brojevi.keras')
img = Image.open('./LV8/test.png').convert('L')

#skalirano od 0-1
img_array = np.array(img, dtype='float32') / 255
print(img_array.shape)

#pretvorila u vektor da bi uslo u mrezu da predict-a
image_s = img_array.reshape(1, 28*28)
print(image_s.shape)

predictions = model.predict(image_s)
#izbacuje index najveceg elementa, index predstavlja prediktan broj
#bolje je ovako nego ograniciti na >=0.5
print(np.argmax(predictions))
