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
On a rajouté à notre RGB image (3 channels) 2 nouveaux channels qui apportent des informations en plus au modèle.

## Segmentation
La segmentation (classification) est l'utilisation d'un modèle pré-entrainé pour reconnaitre les différentes zones d'un visage humain. Ces informations sont renvoyées sous forme de tensor de même taille que l'image et sur 1 seule couche que l'on concaténe avec l'image à impainter. Chaque pixel possède une valeur symbolisant une zone probable.

La segmentation est très puissante et à grandement amélioré les résultats que l'on avait jusque là. La qualité de la segmentation est bien meilleure à plus haute résolution, nous utilisons donc une interpolation et un maxpool pour améliorer les résultats. Cependant ils ne sont quand même pas toujours excellent et améliorer la segmentation pourrait être un très bon moyen d'avoir de meilleurs résultats.

Attention cependant, si vous refaites la segmentation à ne pas donner au modèle trop d'information. Ce qu'on faisait initialement était de reprendre les 3 couches très détaillées sorties par le segmenteur ce qui recréait les images à l'identique quelque soit les zones impaintés.

A savoir aussi que nous avons, dans notre version finale, utilisé une version simplifié de la segmentation en réduisant le nombre de features reconnues. L'objectif était d'avoir moins de choses à mémoriser du point de vue du modèle. Cependant le modèle apprend quand même à reconnaitre ces features avec une absence d'indication (exemple : un "vide" autour des yeux sera reconstruit comme des lunettes). Cela ne semble donc pas économiser quoi que ce soit du point de vue du modèle. Il pourrait donc être pertinent d'enlever la simplification pour permettre au modèle d'avoir plus d'informations.

## Keypoints
On a ici aussi utilisé un modèle pré-entrainé qui, lorsqu'on lui donne une image en entrée nous sort une liste de points clés du visage. On convertie ensuite cette liste de points en un tensor de même dimension que l'image et sur 1 seule couche où les pixels indiqués par les coordonnés de keypoints sont en blanc et le reste en noir. 

Le modèle marche très bien et l'ajout des keypoints améliore les résultats obtenus. Cependant les keypoints ne fonctionnant pas à basse résolution nous sommes du coup obligés d'entrainer directement sur des images en 128x128 ce qui est plus long.

## Super Résolution
On a utilisé ici un ESRGAN pré-entrainé qui sert à améliorer la qualité d'une image donné. L'idée était que le modèle ne pouvant apprendre à reconstuire des images que de qualité limité, la reconstruction aurait toujours une plus petite définition que l'originale et il aurait été intéressant de regagner en qualité. 

Le modèle marche très bien et donne de bons résultats sur des images classiques mais la super résolution d'images reconstruire crée des artefacts étranges et du flou au niveau des zones impainté. Il serait du coup peut être intéressant d'entrainer un modèle de super résolution nous même qui saurait gérer ce problème (ou en trouver un qui marche sur internet).

# Loss
## Perceptuelle
La loss perceptuelle est celle qui donne le meilleur résultat pour l'instant. Nous avons tout d'abord utilisé celle d'un modèle VGG16 pré-entrainé et les résultats donnés étaient vraiment bon. Il est nécessaire de la coupler à une loss TotalVar et à une loss L1.

Cependant VGG16 étant entrainé sur des images génériques nous avons ensuite essayé de développer un modèle entrainé spécifiquement sur des visages pour en extraire un vecteur de features censé être adapté au problème. Pour cela nous avons entrainé un auto-encoder à encoder puis décoder des images. Nous n'utilisons par la suite que la fonction d'encodage pour obtenir le vecteur de features sur lequel et effectuons une MSE entre lui et l'input. 

Cette loss perceptuelle d'auto encoder est une bonne piste mais ne donne des résultats flou. Il serait sans doute possible d'obtenir une meilleure loss avec un auto-encoder plus "profond" dans lequel on extraierait des features à différent niveau.

## Keypoints / Segmentation
Nous avons esayé de reprendre les outils de segmentation et de keypoints pour en faire des Loss (en comparant les informations de l'image d'origine et de l'image prédite) mais les résultats étaient à chaque fois équivalents ou moins bon quel que soit le poids accordé à ces 2 loss.

# Data
## Datasets
Nous avons commencé avec LFW puis rajouté Flickr et avons gardé ces deux là pendant un moment. A la toute fin du stage nous avons rajouté celeba et UTK.

## Normalisation
On en faisait au début mais on a arrêté quand on est passé sur du 5 channels en input. Ca peut cependant être une piste d'essayer d'en remettre auxquel cas il faudrait utiliser la même que celle de VGG16. Ces normalisations sont disponibles dans le module data.py.

## Data Augmentation
On a essayé la data augmentation (zoom, mirroir, rotation..) mais les résultats n'étaient pas transcendants. Pire les zooms réduisent souvent la qualité de la segmentation ce qui fait perdre des informations au modèle.

## Masques 
Les premiers masques étaients des carrés de taille et de position aléatoire. Ils étaient très rapide à process mais n'étaient pas très pertinents. On a donc ensuite utilisé des masques crées par nvidia qui étaient plus naturels. Nous avons ainsi eu de bien meilleurs résultats.

Une approche proposé par David en fin de stage était de commencer dans un premier temps par des très gros masques (qui recouvriraient l'intégralité de l'image) pour apprendre à reconstruire "grossierement" puis affiner avec des masques irréguliers de Nvidia.

# Pistes non explorés (du plus au moins intéressant)
## Modèles de diffusion
Le stage à eu lieu au moment où les modèles de diffusion ont vraiment commencé à exploser (avec Dalle-2, [RePaint ](https://github.com/andreas128/RePaint), [HuggingFace](https://github.com/huggingface/diffusers)...). Ils sont hyper puissants pour créer des choses à partir de rien mais on a eu un peu plus de mal à les faire fonctionner en tant qu'impainter. Par exemple avec RePaint  on a pu reconstruire un nez d'excellente qualité mais qui faisait la moitié du visage.

Il aurait donc été intéressant d'essayer d'entrainer nous même des modèles de diffusion mais comme ces modèles sont relativement récents et qu'on arrivait en fin de stage nous n'avons pas eu le temps d'avoir des entrainements concluants. C'est en tout cas la piste la plus prometteuse.

## Deepfillv2
On a testé des modèles type [DeepFillv2](https://github.com/nipponjo/deepfillv2-pytorch) mais ça ne marchait pas très bien.

# Application
Au début on voulait utiliser Gradio mais c'était assez limité et on aurait pas pu implémenter la segmentation et les keypoints. Donc on a développé rapidement une appli web mais comme c'était pas l'objet du stage initialement l'appli est un peu bâclé et sera assez dur à reprendre ensuite. Elle utilise python pour le backend et du html/js/css pur pour le front.
