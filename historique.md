Historique des pistes abordées lors de ce stage :

# Modèles
## UNet 
Les modèles de types UNet sont ceux qui nous ont donnés les meilleurs résultats
jusque là. Nous avons testé avec différents types de convolution et les meilleurs résultats que nous  
avons obtenus est de mettre une convolution partielle pour toutes les couches sauf pour la dernière  
où on met une convolution classique. La gated et diluted convolution n'ont pas donné de meilleurs résultats.

Les UNet utilisés pendant le stage ayant peu de couche il serait intéressant de tester ce que donnerait un UNet plus "robuste" avec plus de couches. Ce choix d'augmenter le nombre de couche à été implémenté et il est possible de créer un UNet plus costaud avec le paramètre **doubleLayer=True**.

## Pixel Shuffle
Le pixel shuffle n’était pas très efficace pour l’impainting mais est cependant très utile pour  
l’augmentation de résolution.

## Modèles consécutifs
La méthode retenue pendant un moment était de combiner 2 modèles avec des rôles différents : le premier, un impainter, remplacait la zone définie et le deuxième, un augmenter, qui améliorait
le résultat pour le débruiter. Nous avons pour cela utilisé les deux modèles précédemment énoncé.

Seulement lorsque nous sommes passé de 3 à 5 couches en input (avec les keypoints et la segmentation) nous avons mis de côté l'augmenter et avons seulement gardé un UNet. Je ne me rappelle plus vraiment si il y avait une bonne raison d'écarter l'augmenter mais il serait peut être intéressant de tester de le rajouter.

## GAN
Le modèle que nous avons implémenté et testé était relativement simpliste donc les résultats n'étaient pas terrible. Au bout de très peu d'itération le GAN apprenait des "combines" pour améliorer sa loss sans améliorer la qualité de reconstruction (changement de teinte de l'image). 

Les pistes envisagés étaient  de continuer sur le patchgan et de demander de l'aide à des spécialistes de GAN parce que c'est vraiment dur à entrainer. 

# Components 
## Segmentation
La segmentation (classification) est l'utilisation d'un modèle pré-entrainé pour reconnaitre les différentes zones d'un visage humain. Ces informations sont renvoyées sous forme de tensor de même taille que l'image et sur 1 seule couche que l'on concaténe avec l'image à impainter. Chaque pixel possède une valeur symbolisant une zone probable.

La segmentation est très puissante et à grandement amélioré les résultats que l'on avait jusque là. La qualité de la segmentation est bien meilleure à plus haute résolution, nous utilisons donc une interpolation et un maxpool pour améliorer les résultats. Cependant ils ne sont quand même pas toujours excellent et améliorer la segmentation pourrait être un très bon moyen d'avoir de meilleurs résultats.

Attention cependant, si vous refaites la segmentation à ne pas donner au modèle trop d'information. Ce qu'on faisait initialement était de reprendre les 3 couches très détaillées sorties par le segmenteur ce qui recréait les images à l'identique quelque soit les zones impaintés.

## Keypoints
## Super Résolution

# Loss
## Perceptuelle
La loss perceptuelle est celle qui donne le meilleur résultat pour l'instant. Nous avons tout d'abord utilisé celle d'un modèle VGG16 pré-entrainé et les résultats donnés étaient vraiment bon. Il est nécessaire de la coupler à une loss TotalVar et à une loss L1.

Cependant VGG16 étant entrainé sur des images génériques nous avons ensuite essayé de développer un modèle entrainé spécifiquement sur des visages pour en extraire un vecteur de features censé être adapté au problème. Pour cela nous avons entrainé un auto-encoder à encoder puis décoder des images. Nous n'utilisons par la suite que la fonction d'encodage pour obtenir le vecteur de features sur lequel et effectuons une MSE entre lui et l'input. 

Cette loss perceptuelle d'auto encoder est une bonne piste mais ne donne des résultats flou. Il serait sans doute possible d'obtenir une meilleure loss avec un auto-encoder plus "profond" dans lequel on extraierait des features à différent niveau.

## Keypoints / Segmentation
Nous avons esayé de reprendre les outils de segmentation et de keypoints pour en faire des Loss (en comparant les informations de l'image d'origine et de l'image prédite) mais les résultats étaient à chaque fois équivalents ou moins bon quel que soit le poids accordé à ces 2 loss.

# Data
## Datasets
## Normalisation
## Data Augmentation
## Masques 

# Autre
## Augmentation graduelle de la résolution

# Pistes non explorés (du plus au moins intéressant)
## Modèles de diffusion
## Deepfillv2

# Conclusion

# Application
Au début on voulait utiliser Gradio mais c'était assez limité et on aurait pas pu implémenter la segmentation et les keypoints. Donc on a développé rapidement une appli web mais comme c'était pas l'objet du stage initialement l'appli est un peu bâclé et sera assez dur à reprendre ensuite. Elle utilise python pour le backend et du html/js/css pur pour le front.
