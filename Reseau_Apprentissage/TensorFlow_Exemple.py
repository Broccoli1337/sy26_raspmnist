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

#Donne la dimension d'une array
#train_images.shape

#Donne le nb d'elements dans une array
#len(train_labels)

#train_labels

#test_images.shape

#len(test_labels)

#Permet d'afficher une image avec un code couleur différent en fonction de la valeur de chaque pixel

#plt.figure()
#plt.imshow(train_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.plot()


#On modifie les valeurs pour qu'elles se situent entre 0 et 1
#Au lieu d'etre entre 0 et 255
train_images = train_images / 255.0

test_images = test_images / 255.0

#Affiche le set de training, sous la forme d'images avec leur classe indiqué en dessous

#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])


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


#def plot_image(i, predictions_array, true_label, img):
#  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
#  plt.grid(False)
#  plt.xticks([])
#  plt.yticks([])
#  
#  plt.imshow(img, cmap=plt.cm.binary)
#
#  predicted_label = np.argmax(predictions_array)
#  if predicted_label == true_label:
#    color = 'blue'
#  else:
#    color = 'red'
#  
#  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                               100*np.max(predictions_array),
#                                class_names[true_label]),
#                                color=color)

#def plot_value_array(i, predictions_array, true_label):
#  predictions_array, true_label = predictions_array[i], true_label[i]
#  plt.grid(False)
#  plt.xticks([])
#  plt.yticks([])
#  thisplot = plt.bar(range(10), predictions_array, color="#777777")
#  plt.ylim([0, 1]) 
#  predicted_label = np.argmax(predictions_array)
# 
#  thisplot[predicted_label].set_color('red')
#  thisplot[true_label].set_color('blue')
    
#i = 0
#plt.figure(figsize=(6,3))
#plt.subplot(1,2,1)
#plot_image(i, predictions, test_labels, test_images)
#plt.subplot(1,2,2)
#plot_value_array(i, predictions,  test_labels)

#i = 12
#plt.figure(figsize=(6,3))
#plt.subplot(1,2,1)
#plot_image(i, predictions, test_labels, test_images)
#plt.subplot(1,2,2)
#plot_value_array(i, predictions,  test_labels)

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
#num_rows = 5
#num_cols = 3
#num_images = num_rows*num_cols
#plt.figure(figsize=(2*2*num_cols, 2*num_rows))
#for i in range(num_images):
#  plt.subplot(num_rows, 2*num_cols, 2*i+1)
#  plot_image(i, predictions, test_labels, test_images)
#  plt.subplot(num_rows, 2*num_cols, 2*i+2)
#  plot_value_array(i, predictions, test_labels)
    
# Prend une image du groupe de test pour effectuer une prediction
img = test_images[0]

#print(img.shape)

# Ajout de l'image au groupe d'image dont on veut prédire les classes
img = (np.expand_dims(img,0))

print(img.shape)

#Prediction de la classe de l'image
predictions_single = model.predict(img)

#print(predictions_single)

#plot_value_array(0, predictions_single, test_labels)
#_ = plt.xticks(range(10), class_names, rotation=45)

#Donne la classe qui est la plus "probable" pour l'image 0
#np.argmax(predictions_single[0])