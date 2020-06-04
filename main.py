import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(training_images, training_layers), (testing_images, testing_layers) = datasets.cifar10.load_data()
training_images, testing_images = training_images/255, testing_images/255
class_names = ['Plane', 'Car', 'Bird','Cat', 'Deer', 'Dog', 'Frog','Horse', 'Ship', 'Truck']

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[training_layers[i][0]])
plt.show()
training_images = training_images[:20000]
training_layers = training_layers[:20000]
testing_images = testing_images[:4000]
testing_layers = testing_layers[:4000]

model = models.load_model('image_classifier.model')
img = cv.imread('horse.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img])/255)
index = np.argmax(prediction)
print(f"Prediction is {class_names[index]}")

