# Classification Pathologique Vocale par Apprentissage Automatique

Ce dépôt implémente un pipeline complet de classification des pathologies laryngées à partir d'enregistrements vocaux, utilisant des modèles transformeurs pré-entraînés sur la parole et des architectures de fusion multi-branche. Le travail est axé sur des données cliniques réelles issues de l'Hôpital Foch, complétées par la base Saarbruecken (SVD) comme groupe contrôle.

---

## Présentation générale

L'objectif est de développer et comparer différentes approches de classification vocale pathologique en contexte hospitalier, face à des données de taille limitée et potentiellement bruitées. Le projet progresse selon une séquence d'expériences :

1. Benchmark de 9 architectures SSL (Self-Supervised Learning) pour la classification binaire
2. Classification multi-classes (baseline sans augmentation)
3. Augmentation de données rigoureuse avec validation acoustique
4. Fusion bi-branche avec validation croisée 5-plis

---

## Jeux de données

### Sources

- **FOCH (pathologiques)** : 287 enregistrements issus de la clinique ORL, voyelles tenues segmentées en répertoire plat. Métadonnées cliniques dans `orl_df_vowel_master.csv` (pathologie, patient, etc.).
- **Saarbruecken (SVD, sains)** : 687 enregistrements de base de référence, groupe contrôle. Seule la voyelle /a/ à hauteur habituelle (`a_h`) est retenue pour cohérence acoustique avec les enregistrements FOCH.

### Prétraitement et équilibrage

1. **Nettoyage des métadonnées** : normalisation des libellés pathologiques (gestion des espaces parasites, casse hétérogène, valeurs vides).
2. **Déduplication patient** : un seul enregistrement conservé par patient (clé `Last_Name`) pour éviter la fuite d'information entre jeux d'entraînement et validation.
3. **Équilibrage des classes** : sous-échantillonnage du groupe contrôle (SVD) pour obtenir des effectifs comparables par classe, prévenant l'artefact d'une performance dominée par la classe majoritaire.

### Classes retenues

- **PR** (Presbylarynx) — pathologie vocale liée à l'âge
- **LMB** (Lésion laryngée) — lésion bénigne du larynx
- **Fuite glottique** — dysphonie par fermeture incomplète
- **Healthy** — groupe contrôle sain (Saarbruecken)

La classe Cancer est exclue (support insuffisant, ~26 échantillons).

---

## Démarche expérimentale : quatre étapes

### 1. Benchmark binaire (1_0_foch_test.ipynb)

Classification **sain vs. pathologique** comparant 9 architectures SSL :

| Modèle | Paramètres | Type d'entrée | Val AUC | Temps (approx) |
|--------|-----------|---------------|---------|----------------|
| WavLM-Large | 316 M | Waveform | 0.999 | 13 min |
| Wav2Vec2-Large-Robust | 317 M | Waveform | 0.998 | 8 min |
| MERT-v1-330M | 330 M | Waveform | 0.997 | 9 min |
| WavLM-Base | 94 M | Waveform | 0.995 | 3 min |
| Whisper-Base | 74 M | Spectrogram | 0.995 | 4 min |
| Wav2Vec2-Base | 95 M | Waveform | 0.993 | 5 min |
| AST | 86 M | Spectrogram | 0.988 | 4 min |
| HuBERT-Base | 94 M | Waveform | 0.972 | 3 min |

**Approche** : chaque modèle est entraîné indépendamment, le backbone CNN/patch-embedding est gelé (extracteur de features pré-entraîné), seule la tête de classification est fine-tunée. **WavLM-Large** émerge comme meilleur modèle (AUC = 0.999).

**Optimisations appliquées** :
- Précision mixte automatique (AMP) : ~2-3x plus rapide, déduit la VRAM
- Accumulation de gradient : batch effectif 16 avec VRAM limée
- Planificateur cosinus avec échauffement : convergence plus stable
- Early stopping (patience=5) : prévient le sur-apprentissage

### 2. Baseline multi-classes (2_0_foch_test.ipynb)

Classification **quatre classes** (PR, LMB, fuite glottique, Healthy) avec **WavLM-Large**, **sans augmentation de données** — groupe de contrôle référence.

**Démarche** :
- Données balancées par sous-échantillonnage du groupe contrôle (~100 sains, ~70 pathologies)
- Split 80/20 stratifié ET groupé par patient (garantit absence de fuite)
- Découpe gelé, transformeur et tête fine-tunés
- Métriques : balanced accuracy, macro-F1, macro AUC (robustes au déséquilibre)

**Résultats typiques** : balanced accuracy ~0.69, macro-F1 ~0.71

### 3. Augmentation de données (3_0_foch_test.ipynb)

Extension du baseline avec **trois niveaux d'augmentation**, évalués par **ablation** :

| Expérience | Contenu | Effet |
|-----------|---------|--------|
| **E0** | Baseline (2_0) | Référence |
| **E1** | + Augmentation signal-level | Bruit blanc, gain, réverbération légère |
| **E2** | + SpecAugment | Masquage temps-fréquence interne WavLM |
| **E3** | + SeedVC (si disponible) | Conversion de voix conditionnée F0 |

**Principes de validation acoustique** :

Chaque augmentation est vérifiée via **parselmouth** (analyse acoustique Praat) : jitter, shimmer, HNR, F0. Les transformations signal-level (bruit, gain, réverbération) sont additives et ne détruisent pas la source glottique. SeedVC (conversion de voix) subit un gate parselmouth : si jitter/shimmer s'effondrent (> 50% de baisse), le tier est rejeté (pathologie effacée = label faux).

**Rééquilibrage des données** : chaque classe pathologique est complétée par des variantes synthétiques jusqu'à ~80 exemples par classe. Le groupe contrôle est complété par des vrais locuteurs SVD supplémentaires.

**Garanties méthodologiques** :
- Validation **100% originale et fixe** (strictement non augmentée)
- Augmentation **train-only**, post-split
- Traçabilité complète : manifestes CSV décrivant provenance (`is_aug`, `source_filepath`)
- Garde-fous anti-fuite : aucun patient source augmenté en validation

### 4. Fusion bi-branche avec validation croisée (4_0_foch_test.ipynb)

Architecture **multi-branche complexe** (WavLM-Large + ResNet101), évaluée par **StratifiedKFold à 5 plis**, avec **augmentation injectée pli par pli**.

**Architecture** :

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Audio (WAV)                         │
└──────────────┬──────────────────────────┬──────────────────┘
               │                          │
        ┌──────▼──────┐          ┌────────▼────────┐
        │   Branche A │          │   Branche B     │
        │ (WavLM-Lge) │          │  (ResNet101)    │
        │   1024-d    │          │   2048-d        │
        └──────┬──────┘          └────────┬────────┘
               │                          │
               │         Fusion           │
               └──────────┬───────────────┘
                          │
                  ┌───────▼────────┐
                  │ Tête de classe │
                  │  3072 → 512    │
                  │  512 → 4       │
                  └────────────────┘
```

**Branche A (microscopique)** : WavLM-Large gelé, embeddings 1024-d via moyenne temporelle. Capture : jitter, shimmer, apériodicité court-terme.

**Branche B (macroscopique)** : ResNet101 pré-entraîné ImageNet sur spectrogramme Mel 128-bin. Le Mel mono-canal est répliqué sur 3 canaux pour exploiter la première convolution. Capture : étalement formantique, bandes de bruit, perte de structures harmoniques.

**Validation croisée** : StratifiedKFold(n_splits=5, shuffle=True). À chaque pli :
- Train reçoit originaux + segments augmentés (hors patients en validation)
- Validation reste **strictement originale**
- Modèle réinitialisé à chaque pli
- Assertions garantissent l'absence de fuite patient

**Résultats typiques** : OOF balanced accuracy ~0.65, macro-F1 ~0.65, macro AUC ~0.85. Dispersion entre plis (±0.07 pour balanced accuracy) indique une stabilité modérée.

---

## Modèles et architectures

### Sélection d'architectures (1_0)

**Transformeurs sur waveform brute** (CNN feature extractor gelé) :
- Wav2Vec2 (Facebook) : contrastive SSL
- WavLM (Microsoft) : SSL avec débruitage
- HuBERT (Facebook) : masked language model on audio
- MERT (Music-aware) : pré-training music, 24 kHz requis

**Transformeurs sur spectrogram** (patch embedding gelé) :
- AST (MIT) : ViT pré-entraîné AudioSet
- Whisper (OpenAI) : ASR encoder repurposé

### Fine-tuning stratégies

**Stratégie 1 (baseline, 2_0)** :
- CNN/patch-embedding gelé (extracteur pré-entraîné fixe)
- Transformer et tête fine-tunés
- Réduit overfitting, économise VRAM

**Stratégie 2 (fusion, 4_0)** :
- Branche audio (WavLM) complètement gelée
- Branche vision (ResNet) pré-entraînée, fine-tunée
- Tête de fusion entraînable
- 12% des paramètres entraînables (44M/359M total)

---

## Configuration de l'environnement

### Dépendances

```
Python >= 3.10
torch >= 2.0
torchaudio
transformers >= 4.36
torchvision (pour ResNet101)
scikit-learn
pandas, numpy
matplotlib, seaborn
soundfile (pour lecture/écriture audio augmenté)
parselmouth (pour validation acoustique)
ffmpeg (backend torchaudio, doit être dans PATH)
```

### Chemins critiques

```
C:/Users/AdminIA/Documents/models/
  ├── wavlm-large/                    # Backbone WavLM local
  ├── .poids_modèles/                 # Poids entraînés (sauvegarde)
  └── .foch_métriques/                # Métriques, matrices, graphes

C:/Users/AdminIA/Desktop/ORL_IA_FOCH_Callum_HOLLIDAY/
  ├── csv_best/orl_df_vowel_master.csv         # Métadonnées FOCH
  ├── Segmented_Voyelles_Best/                 # Enregistrements FOCH
  └── Segmented_Voyelles_Augmented/            # Données augmentées
      ├── signal/                               # Variantes signal-level
      ├── vc/                                   # Conversions SeedVC
      ├── manifest_train_bal.csv                # Manifeste augmenté (signal)
      └── manifest_train_bal_vc.csv             # Manifeste augmenté (full)

C:/Users/AdminIA/Documents/SVD/healthy/        # Base Saarbruecken
  └── <patient_id>/vowels/*a_h.wav
```

### Initialisation FFmpeg

```python
import shutil, os
_ffmpeg_bin = r'C:/Users/AdminIA/Downloads/ffmpeg-8.1.1-full_build-shared/ffmpeg-8.1.1-full_build-shared/bin'
os.environ['PATH'] += os.pathsep + _ffmpeg_bin
print(shutil.which("ffmpeg"))  # Vérifier présence
```

---

## Guide d'exécution

### Exécution complète (4 notebooks)

```
1. Ouvrir 1_0_foch_test.ipynb
   → Cellules 1-8 : benchmark binaire complet (9 architectures)
   → Durée : ~1-2 heures (GPU CUDA 8 Go)
   → Sortie : classement des modèles, métriques

2. Ouvrir 2_0_foch_test.ipynb
   → Cellules 1-8 : classification multi-classes baseline
   → Durée : ~20-30 min
   → Sortie : poids meilleur modèle, dashboard d'évaluation

3. Ouvrir 3_0_foch_test.ipynb
   → Cellules 1-8 : augmentation de données + ablation
   → Durée : ~60-90 min (3 expériences E1-E3)
   → Sortie : variantes augmentées, comparaison ablation

4. Ouvrir 4_0_foch_test.ipynb
   → Cellules 1-8 : fusion bi-branche + 5-fold CV
   → Durée : ~2-3 heures (5 modèles WavLM + ResNet101)
   → Sortie : métriques OOF agrégées, dispersion par pli
```

### Points critiques

**Avant de lancer** :
- Vérifier `shutil.which("ffmpeg")` (backend torchaudio)
- Télécharger les modèles HuggingFace localement (notebook 1 fournit les `huggingface-cli` commands)
- Vérifier l'espace disque (~50 GB pour modèles + augmentés)

**Durant l'entraînement** :
- GPU : 8 Go VRAM minimum (WavLM-Large + ResNet101 consomme ~7 Go)
- CPU : multi-threading autorisé (`num_workers=0` sur Windows pour éviter deadlock Jupyter)
- Early stopping activé : arrêt automatique si validation ne s'améliore pas (patience=5 ou 6)

**Après chaque notebook** :
- Vérifier les fichiers de sortie dans `.poids_modèles` et `.foch_métriques`
- Les PNG dashboards visualisent confusions, ROC, courbes d'apprentissage
- Les JSON métriques archivt les scores (pour comparaison ultérieure)

---

## Structure du dépôt

```
Classification_Algo/
├── README.md                          # Ce fichier
├── 1_0_foch_test.ipynb               # Benchmark binaire (9 SSL architectures)
├── 2_0_foch_test.ipynb               # Baseline multi-classes
├── 3_0_foch_test.ipynb               # Augmentation + ablation
├── 4_0_foch_test.ipynb               # Fusion bi-branche + 5-fold CV
└── SVD.ipynb                         # Utilities pour dataset Saarbruecken (optionnel)
```

---

## Approche méthodologique générale

### Robustesse aux données limitées

1. **Architectures pré-entraînées** : exploitation du transfert learning (backbone SSL sur 960h LibriSpeech, 53k AudioSet, etc.)
2. **Feature extractor gelé** : les couches CNN bas-niveau ne sont pas fine-tunées (réduisent le sur-apprentissage, économisent VRAM)
3. **Augmentation rigoureuse** : transformations signal-level et TF-domain, validées acoustiquement (parselmouth)
4. **Validation croisée** : spread du risque sur K plis, intervalles de confiance

### Prévention des fuites et biais

- **Déduplication patient** : un enregistrement par patient (clé `Last_Name`)
- **Split groupé** : aucun patient ne peut être en train et validation simultanément
- **Augmentation train-only** : segemnts synthétiques injectés uniquement dans le train de chaque pli, en écartant les patients du pli de validation
- **Assertions explicites** : vérifications systématiques d'absence de fuite en fin de chaque cellule de split

### Métriques adaptées au déséquilibre

- **Balanced accuracy** : moyenne des rappels par classe (robust à déséquilibre)
- **Macro-F1** : F1 par classe, puis moyenne (traite chaque pathologie à parité)
- **Macro AUC (OvR)** : area under curve en configuration un-contre-tous, moyenne

---

## Résultats synthétiques

### Benchmark binaire (Healthy vs. Pathological, 1_0)

**Meilleur modèle** : WavLM-Large
- Val AUC : 0.999
- Validation Accuracy : 99%
- Temps d'entraînement : ~13 minutes

### Baseline multi-classes (2_0)

**WavLM-Large, validation split unique**
- Balanced Accuracy : ~0.69
- Macro-F1 : ~0.71
- Macro AUC : ~0.88

### Augmentation (3_0)

**E1 (signal-level)** : Balanced Acc ~0.73, Macro-F1 ~0.74
**E2 (+ SpecAugment)** : Balanced Acc ~0.75, Macro-F1 ~0.77
**E3 (+ SeedVC)** : Dépend de disponibilité SeedVC (tier souvent rejeté par gate parselmouth)

### Fusion bi-branche 5-fold (4_0)

**OOF agrégé** :
- Balanced Accuracy : 0.646 ± 0.075
- Macro-F1 : 0.649 ± 0.088
- Macro AUC : 0.855 ± 0.041

Dispersion notable indique que la fusion multi-branche bénéficie d'une validation croisée rigoureuse, mais reste sensible à la stratification par pli.

---

## Pistes d'amélioration

1. **Dégel progressif** : débuter gelé, puis unfreezing des top-K couches transformer (notebook 4_0 entame cette démarche)
2. **SpecAugment avancé** : masquage temps-fréquence plus agressif (TimeStretch, PitchShift)
3. **Pondération de fusion** : remplacer concaténation simple par attention apprendre (CrossAttention entre branches)
4. **Ensembling** : agrégation des K modèles de plis 4_0 pour inférence (moyenne des probabilités)
5. **Audit données** : vérifier homonymes (ex. MARTIN ×7) via `patient_id` clinique, pas juste `Last_Name`
6. **Données externes** : intégration d'autres bases pathologie vocale (VoxCeleb pathologique, corpus multilingues)

---

## Références et citations

- **WavLM** : Chen, S., et al. (2022). "WavLM: Large-Scale Self-Supervised Pre-Training for Speech"
- **Wav2Vec 2.0** : Baevski, A., et al. (2020). "wav2vec 2.0"
- **HuBERT** : Hsu, W.-N., et al. (2021). "HuBERT"
- **AST** : Gong, Y., et al. (2021). "AST: Audio Spectrogram Transformer"
- **Whisper** : Radford, A., et al. (2022). "Robust Speech Recognition via Large-Scale Weak Supervision"

---

## Contact et notes

Ce pipeline a été développé dans un contexte de recherche clinique à l'Hôpital Foch. Les données restent confidentielles (données cliniques réelles). La reproduction complète du pipeline sur données synthétiques est possible ; seule la performance absolutebénéficie des véritables enregistrements pathologiques.

Pour questions ou améliorations : vérifier les cellules de documentation markdown dans chaque notebook pour détails techniques complets.
