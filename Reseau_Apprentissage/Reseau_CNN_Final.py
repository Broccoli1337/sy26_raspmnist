#----------------------- Initialisation du script -----------------------
from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
from PIL import Image
import numpy as np

#import cv2
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

#Installation et import des dépendances de TenserFlow

import os

#pip install --user png
#https://github.com/drj11/pypng/
#https://pythonhosted.org/pypng/ex.htmlhttps://pythonhosted.org/pypng/ex.html
#import png

#import scipy.misc

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
imgDir = "./imgData/"
#Chemin de l'image a tester
imgPath = "./imgData/sample.png"

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

#Traitement d'une image pour qu'elle soit au meme format que celles de la base de donnée utilisée
# ------------------------ Debut convertImage(img_path)
def convertImage(img_path):
    
    size = 28
    
    img = Image.open(img_path) #Ouverture de l'image
    img = np.asarray(img)
    img.setflags(write=1)
    img = negative(img)
    img = Image.fromarray(img)
    
    wpercent = (size/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    #https://stackoverflow.com/questions/273946/how-do-i-resize-an-image-using-pil-and-maintain-its-aspect-ratio
    img = img.resize((size,hsize+1),Image.ANTIALIAS) #Redimensionne l'image
    img = img.convert("L") #Conversion des couleurs en niveaux de gris
    img.save(imgDir + "new_sample.png") #Sauvegarde de l'image
    
# ------------------------ Fin convertImage(img_path)   

#Copier de StackOverflow
#https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def negative(img):
    
    height = len(img)
    width = len(img[0])
    negImg = img
    
    for ligne in range(height):
        for col in range(width):
            r = 255 - img[ligne][col][0]
            g = 255 - img[ligne][col][1]
            b = 255 - img[ligne][col][2]
            negImg[ligne][col]=[r,g,b]
            
    return negImg

#Prediction de la classe de l'image
# ------------------------ Debut predict_img()
def predict_img(img_path):

    #Provisoire
    #img_path = test_images[1]    
    
    img = Image.open(img_path)
    img = np.asarray(img)
    img = img / 255.0
    
    #img = test_images[8]
    newImg = Image.fromarray(img*255)
    
    # Ajout de l'image au groupe d'image dont on veut prédire les classes
    img = (np.expand_dims(img,0))
    
    prediction = model.predict(img)
    prediction = np.argmax(prediction[0])

    #print("Valeur attendue : ",test_labels[1])
    print("Prediction : ",class_names[prediction])

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
    
    convertImage(imgPath)
    prediction = predict_img(imgDir + "new_sample.png")
else:
    makeSave()