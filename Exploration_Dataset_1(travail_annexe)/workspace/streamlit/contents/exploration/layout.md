## Compréhension du Layout (GT)

Le fichier XML GT fournit une segmentation spatiale de chaque facture.  
Chaque segment contient :
- une bounding box (x, y, largeur, hauteur)  
- un label ou une catégorie  
- une position dans la structure du document  

---

### Ce que le GT révèle

- La facture est découpée en zones logiques (totaux, TVA, adresse, lignes d’articles…).  
- Le GT sert de base au modèle OCR pour associer chaque zone à un texte.  
- La structure spatiale est cohérente et répétitive, car le dataset est homogène.

---

### Visualisation GT

Nous avons superposé les bounding boxes GT sur les images pour :
- valider la cohérence des segments  
- comprendre la logique de découpage  
- identifier les zones clés du document  

Cette étape a permis de confirmer que le GT est fiable et bien structuré.
