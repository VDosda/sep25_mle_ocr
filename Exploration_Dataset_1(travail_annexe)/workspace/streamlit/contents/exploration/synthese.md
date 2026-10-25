## Apport du dataset

L’exploration du dataset nous a permis de comprendre :
- la structure GT (segmentation spatiale)  
- la qualité et la logique de l’OCR  
- la cohérence des indices de confiance  
- la qualité homogène des images  

---

### Limites du dataset

Le dataset ne permet pas la classification car :
- il contient uniquement des factures  
- il n’y a aucune diversité de documents  
- les structures sont trop similaires  

---

### Pistes pour aller plus loin

- utiliser les segments OCR fiables pour entraîner un modèle Vision‑Langage  
- combiner image + texte + layout pour créer des embeddings multimodaux  
- rechercher un dataset plus varié pour la classification  

---

### Conclusion

Ce dataset est idéal pour :
- comprendre un pipeline GT → OCR  
- analyser la structure d’un document  
- explorer la qualité des images  

Mais il est insuffisant pour entraîner un modèle de classification robuste.
