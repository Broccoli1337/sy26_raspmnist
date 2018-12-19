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

#---------------------------- Variables
nb_epoch = 25
nb_apprentissage = 5
interval_sauvegarde = 5

#Chemin des sauvegardes
checkpoint_path = "training_final/save.ckpt"
checkpoint_dir = "./" + checkpoint_path;
save_path = "save/network_data.index"
save_dir = "./" + save_path

#Repertoire qui contiendra les images a tester
imgDir = "./images/"
#Chemin de l'image a tester
imgPath = "./image/test.png"

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

# Creation du reseau(apprentissage et sauvegarde)
# ------------------------ Debut create_model()
def makeSave():
    print("MakeSave() ==> Apprentissage en cours")
    
    #Creation du modele
    model = create_model()
    model.summary() #Donne un aperçu du modele

    #------------ Creation d'une fonction de callback pour le chekpoint
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        #Sauvegarde toutes les 5 epochs
        period=interval_sauvegarde)

    #Debut de l'apprentissage
    #epochs = nombre de passages (On atteint 95% de positifs aux alentours de 30 epochs)
    model.fit(train_images, train_labels, epochs=nb_epoch,validation_data = (test_images,test_labels),callbacks = [cp_callback])

    #On regarde les performances sur le groupe de test
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Precision du modele avec apprentissage:', test_acc)
    
    # Save the weights
    model.save_weights('./save/network_data')
    print("Sauvegarde des données")
    
    for x in range(nb_apprentissage):
        model = create_model()
        model.load_weights('./save/network_data')
        model.fit(train_images, train_labels, epochs=nb_epoch,validation_data = (test_images,test_labels),callbacks = [cp_callback])
        
        #On regarde les performances sur le groupe de test
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print('Precision du modele avec apprentissage:', test_acc)
        model.save_weights('./save/network_data')
        
    print("Precision modele final :", test_acc)
    print("Fin => MakeSave()")
    
#------------------------ Fin makeSave()

#Lecture des images de test dans un dossier donné
# ------------------------ Debut get_images()
def get_images():
    images = os.listdir(imgDir)
    x = 0
    images_test = []
    
    for img in images:
        images_test.append(imgDir + img)
    
    return images_test
    
#------------------------ Fin get_images()

#Prediction de la classe de l'image
# ------------------------ Debut predict_img()
def predict_img(img_path):
    
    #Provisoire
    img_path = test_images[1];
    # Ajout de l'image au groupe d'image dont on veut prédire les classes
    img_path = (np.expand_dims(img_path,0))
    
    prediction = model.predict(img_path)
    prediction = np.argmax(prediction[0])
    
    print("Valeur attendue : ",test_labels[1])
    print("Prediction : ",prediction)
    
    return prediction
    
#------------------------ Fin predict_img()

#---------------------------------- Reprise depuis le checkpoint

#Si une sauvegarde existe déjà on la charge pour executer le reseau
if os.path.isfile(save_path):
    print("----------------------- Chargement des données : ")
    # Restore the weights
    model = create_model()
    model.load_weights('./save/network_data')
    print("----------------------- Test du reseau")
    loss,acc = model.evaluate(test_images, test_labels)
    print("Modele restauré, précision : {:5.2f}%".format(100*acc))
    
    prediction = predict_img(imgPath)
else:
    makeSave()