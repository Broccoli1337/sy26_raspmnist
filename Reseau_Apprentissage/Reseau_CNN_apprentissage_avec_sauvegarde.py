#----------------------- Initialisation du script -----------------------
from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#Installation et import des dépendances de TenserFlow

import os

print(tf.__version__)

nb_epoch = 10
interval_sauvegarde = 5

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


# Creation du modele
# ------------------------ Debut create_model()
def create_model():
    #Ajout des layers
    #Flatten() => Transforme les array(28,28) en un vecteur(784)
    #Dense(n,type) => Layer connectés avec n= nombre de noeuds
  model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
  
  model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  
  return model
#------------------------ Fin create_model()

#Creation du modele
model = create_model()
model.summary() #Donne un aperçu du modele

#------------ Creation d'un checkpoint
# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "training_result_test/cp.ckpt"
checkpoint_dir = "./" + checkpoint_path;

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    #Sauvegarde toutes les 5 epochs
    period=interval_sauvegarde)

#Debut de l'apprentissage
#epochs = nombre de passages (On atteint 95% de positifs aux alentours de 30 epochs)
model.fit(train_images, train_labels, epochs=nb_epoch,validation_data = (test_images,test_labels),callbacks = [cp_callback])

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

#---------------------------------- Fin Predictions

#---------------------------------- Reprise depuis le checkpoint

#On ajoute les valeurs sauvegardées au modele, et on lui fait tester un groupe d'image
model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc)) #Affiche le taux de reussite

# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "training_results/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#Recupere le dernier checkpoint créer
latest = tf.train.latest_checkpoint(checkpoint_dir)
#latest

model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))