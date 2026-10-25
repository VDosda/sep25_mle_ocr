## Dataset

Nous avons utilisé le jeu de données fourni pour mieux appréhender le sujet.  
Il s’agit de fichiers résultats de segmentation et océrisation avec indice de confiance, réalisés sur des factures uniquement.

Pour chaque image, le dataset contient :
- un fichier image
- un XML contenant les segments GT (Ground Truth)
- un XML contenant les OCR (texte associé à chaque segment)

### Préparation des données

Nous avons traité ces fichiers afin de :
- nettoyer les données
- formater les informations
- charger le tout dans des dataframes

Des transformations ont été appliquées pour permettre une visualisation graphique des informations GT et OCR.

### Visualisations

Nous avons généré :
- un graphe de visualisation de l’image avec les GT et OCR pour valider la compréhension
- une seconde visualisation pour afficher l’indice de confiance des différents documents

### Limites du dataset

Le dataset fourni ne permet pas de faire de la classification car il est limité aux factures.

Le GT semble correspondre à une segmentation spatiale de l’image, utilisée ensuite par un modèle d’océrisation qui produit :
- les OCR
- l’indice de confiance

### Pistes d’exploitation avancée

On aurait pu utiliser les OCR (box + texte) avec un indice de confiance proche de 1 pour entraîner des modèles vision‑langage (VLM) afin d’apprendre la relation entre :
- l’image (vision)
- le texte (langage)
- la structure (layout)

### Analyse des images

Nous avons appliqué :
- du pré‑processing
- de l’extraction de features
- des comparatifs d’images

Deux méthodes d’estimation de la qualité ont été utilisées.  
Les résultats montrent que :
- le dataset est homogène
- les images sont de bonne qualité
- les deux méthodes sont cohérentes
- les scores sont fortement corrélés
