# Classification Pathologique Vocale par Apprentissage Automatique

Ce dépôt documente un pipeline complet de classification binaire des pathologies vocales basé sur l’état de l’art en apprentissage profond pour l’audio, appliqué à un jeu de données clinique (SVD et FOCH) dans un contexte hospitalier à l’Hôpital Foch.

---

## Présentation du projet

L’objectif est de développer, valider et comparer différents modèles transformeurs pour la détection automatique, à partir d’enregistrements vocaux, d’atteintes pathologiques versus sujets sains. Le classifieur différencie entre :
- **Voix saine (Healthy)**
- **Voix pathologique (Pathological)**

Le modèle et les scripts sont optimisés pour des données cliniques réelles, de taille réduite et potentiellement bruitées.

---

## Jeux de données utilisés

- **FOCH** (pathologiques) : répertoire plat de fichiers `.wav`, un fichier par patient diagnostiqué.
- **SVD** (sains) : parcours des fichiers `/healthy/<patient_id>/vowels/*a_h.wav`, pour échantillonner à hauteur et équilibre du groupe pathologique.

Après équilibrage des classes par sous-échantillonnage, chaque classe comporte **281 échantillons**.

---

## Prétraitement et pipeline

1. **Configuration de l’environnement :**  
   - Gestion des chemins pour FFmpeg et modèles locaux.
   - Sélection du modèle à utiliser via un registre de configuration.

2. **Parsing & équilibrage :**  
   - Sous-échantillonnage aléatoire des sujets sains (SVD) pour équilibrer exact.  
   - Fusion dans un DataFrame unique, sauvegardé pour reprises ultérieures.

3. **Dataset personnalisé (PyTorch) :**  
   - Chargement de l’audio, conversion mono, resampling à 16kHz.
   - Extraction de features adaptée au modèle choisi (transformeur sur onde brute ou spectrogramme).

4. **Séparation train/validation :**  
   - Split 80/20, stratifié.
   - Utilisation de DataLoader PyTorch (`num_workers=0` pour compatibilité Windows).
 
---

## Modèles pris en charge

- Plusieurs architectures testables : Wav2Vec2, WavLM, HuBERT, AST, Whisper, MERT.
- Extraction locale obligatoire des poids de modèles depuis HuggingFace pour respecter la confidentialité et la robustesse en clinique.

Pour chaque architecture, la tête de classification est initialisée aléatoirement pour s’adapter à `num_labels = 2`, et seul l’encodeur transformeur est entraîné (extracteur CNN/embedding gelé).

---

## Optimisations d’entraînement

- **AMP (Mixed precision CUDA)** : doublement de la vitesse et réduction de la mémoire vidéo nécessaire (~8Go VRAM suffisent).
- **Accumulation de gradient** : permet une taille de batch effective plus grande sans dépasser la mémoire disponible.
- **Poids AdamW, schedulers cosinus et early stopping** pour prévenir le surapprentissage sur petit dataset.
- **Checkpoints fréquents** pour reprise sans perte de progression.

---

## Résultats typiques

- Entraînement jusqu’à 30 époques avec arrêt automatique après 5 époques sans amélioration de la validation.
- Précision, matrice de confusion, courbes ROC/PR générées automatiquement à chaque exécution.

---

## Dépendances

```
Python ≥ 3.10
torch ≥ 2.0
torchaudio
transformers ≥ 4.36
scikit-learn
pandas
numpy
matplotlib
seaborn
ffmpeg (doit être présent dans le PATH système)
```

Les poids modèles doivent être copiés localement, par exemple dans `C:\Users\AdminIA\Documents\models\wavlm-large`.

---

## Exécution rapide

1. Configurer l’environnement et le chemin des modèles.
2. Exécuter chaque cellule du notebook dans l’ordre pour :
   - Préparer les DataFrames et splits
   - Initialiser le modèle
   - Lancer l’entraînement optimisé
   - Visualiser l’évaluation
3. Les checkpoints et historiques d’entraînement/validation sont sauvegardés pour analyse/offline.

---

## Remarques et perspectives

- Adapté pour travaux de recherche translationnelle en IA clinique vocale, en intégrant des contraintes réelles de terrain (bruit, VRAM limitée, échantillons hétérogènes).
- Possibilité de prolonger vers une classification multiclasses ou d’intégrer d’autres bases voice patho.
