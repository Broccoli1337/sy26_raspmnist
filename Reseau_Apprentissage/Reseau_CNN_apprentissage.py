#----------------------- Initialisation du script -----------------------
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#----------------------- Debut de l'apprentissage ------------------------------

#Import de la base de donnée FashionMNIST
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Definition des classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#On modifie les valeurs pour qu'elles se situent entre 0 et 1
#Au lieu d'etre entre 0 et 255
train_images = train_images / 255.0

test_images = test_images / 255.0

#Ajout des layers
#Flatten() => Transforme les array(28,28) en un vecteur(784)
#Dense(n,type) => Layer connectés avec n= nombre de noeuds
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


#Compilation du modèle
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#Debut de l'apprentissage
#epochs = nombre de passages (On atteint 95% de positifs aux alentours de 30 epochs)
model.fit(train_images, train_labels, epochs=25)

#On regarde les performances sur le groupe de test
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

#---------------------------------- Fin de l'apprentissage

#---------------------------------- Predictions

#On stoke les predictions dans une array
predictions = model.predict(test_images)

#Correspond aux predictions de la premiere image
#Contient une array de 10 elements, chacun donnant la valeur prédit pour la classe associée
#predictions[0]

#Affiche la prediction la plus haute, et donc la classe qui est "validé" par le reseau
#np.argmax(predictions[0])

#Donne l'id de la classe pour l'element 0, ce qui permet de verifier que la prediction est juste
#test_labels[0]
    
# Prend une image du groupe de test pour effectuer une prediction
img = test_images[0]

#print(img.shape)

# Ajout de l'image au groupe d'image dont on veut prédire les classes
img = (np.expand_dims(img,0))

print(img.shape)

#Prediction de la classe de l'image
predictions_single = model.predict(img)

#print(predictions_single)

#Donne la classe qui est la plus "probable" pour l'image 0
#np.argmax(predictions_single[0])