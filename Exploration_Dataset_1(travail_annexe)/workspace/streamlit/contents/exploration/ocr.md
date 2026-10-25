## Analyse OCR

Le fichier XML OCR contient :
- le texte extrait pour chaque segment  
- l’indice de confiance associé  
- la position du texte dans l’image  

---

### qualité OCR

Nous avons observé que :
- l’OCR est globalement de bonne qualité  
- les indices de confiance sont élevés  
- les erreurs sont rares et souvent liées à des artefacts visuels  

---

### Visualisation OCR

La superposition OCR + GT nous a permis de :
- vérifier la correspondance texte / zone  
- repérer les segments mal reconnus  
- analyser les erreurs de segmentation ou d’océrisation  

---

### Potentiel d’exploitation

Les segments avec indice de confiance proche de 1 pourraient servir à :
- entraîner un modèle Vision-Langage (VLM)  
- apprendre la relation entre image, texte et structure  
- construire des embeddings multimodaux
