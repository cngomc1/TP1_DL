# MNIST Classification avec TensorFlow

Ce projet contient un réseau de neurones simple pour classifier les chiffres manuscrits du dataset **MNIST** (0 à 9) en utilisant **TensorFlow/Keras**.

---

## **Description**

Le modèle est un **réseau fully-connected (Dense)** avec régularisation **Dropout** :

* **Couche Dense** : 512 neurones avec activation ReLU
* **Dropout** : 20% des neurones désactivés pendant l’entraînement pour éviter l’overfitting
* **Couche de sortie Dense** : 10 neurones avec activation Softmax pour produire les probabilités de chaque chiffre

Le modèle est entraîné sur 60 000 images d’entraînement et évalué sur 10 000 images de test.


## **Prérequis**

* Python 3.12+ (64 bits recommandé)
* TensorFlow 2.x
* Numpy

Installation via pip :

```bash
pip install tensorflow numpy
```


## **Utilisation**

1. **Cloner le projet**

```bash
git clone https://github.com/cngomc1/TP1_DL.git
cd TP1_DL
```

2. **Exécuter le script Python**

```bash
python train_model.py
```

Le script :

* Charge et normalise les données MNIST
* Crée le modèle Dense + Dropout + Softmax
* Entraîne le modèle sur 5 epochs avec batch_size 128
* Évalue la précision sur les données de test
* Sauvegarde le modèle sous `mnist_model.h5`


## **Explications**

* **Vectorisation** : les images 28×28 sont converties en vecteurs de 784 valeurs pour accélérer les calculs matriciels.
* **Batch processing** : 128 images traitées simultanément pour un entraînement plus rapide et stable.
* **Optimiseur Adam** : convergence plus rapide et stable que la SGD simple.
* **Softmax** : permet de transformer la sortie du modèle en probabilités pour la classification multi-classes.

---

## **Évaluation**

Après entraînement, le script affiche :

```text
Précision sur les données de test: 0.XXXX
```

où `0.XXXX` est la précision finale du modèle sur les images jamais vues.

---

## **Sauvegarde du modèle**

Le modèle entraîné est sauvegardé dans le fichier :

```
mnist_model.h5
```

Tu peux le recharger avec Keras pour faire des prédictions ultérieurement :

```python
from tensorflow import keras
model = keras.models.load_model("mnist_model.h5")
```

