#Installation et import des dépendances de TenserFlow
#!pip install -q h5py pyyaml 

from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

tf.__version__

#------------------------------------ Debut du script 

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# Creation du model (remplacer par le model qu'on utilisera dans le projet)
# ------------------------ Debut create_model()
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  
  model.compile(optimizer='adam', 
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])
  
  return model
#------------------------ Fin create_model()

# Creation du modele
model = create_model()
model.summary()

#------------ Creation d'un checkpoint
checkpoint_path = "training_result_test/cp.ckpt"
checkpoint_dir = "./" + checkpoint_path;

# Creation d'une fonction de callback pour le checkpoint
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)
#Nouveau model qui utilisera notre checkpoint
model = create_model()

model.fit(train_images, train_labels,  epochs = 5, 
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback])  # ajout du checkpoint dans le modele

#Affichage des fichiers du dossier checkpoint
#!ls {checkpoint_dir}

#Nouveau modele sans entrainement
#model = create_model()
#
#loss, acc = model.evaluate(test_images, test_labels)
#print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

#On ajoute les valeurs sauvegardées au modele, et on lui fait tester un groupe d'image
model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc)) #Affiche le taux de reussite

#Nouvelle version du checkpoint
# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    #Sauvegarde toutes les 5 epochs
    period=5)

model = create_model()
model.fit(train_images, train_labels,
          epochs = 5, callbacks = [cp_callback],
          validation_data = (test_images,test_labels),
          verbose=0)

#! ls {checkpoint_dir}

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