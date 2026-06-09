"""
foch_utils.py — Fonctions partagées pour la classification multi-classes FOCH.

Ce module factorise la logique éprouvée du notebook `2_0_foch_test.ipynb`
(construction du jeu, encodage, découpage groupé par patient, Dataset, modèle,
entraînement, évaluation) afin que `3_0_foch_test.ipynb` (augmentation de données)
la réutilise sans duplication ni dérive de comportement.

Le baseline `2_0` reste, lui, volontairement autonome (groupe de contrôle).
"""
import os
import glob
import json
import random

import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════════════════
# 1. Construction du jeu de données de base (réplique exacte de 2_0)
# ═══════════════════════════════════════════════════════════════════════════
def build_base_dataframe(csv_path, foch_audio_dir, svd_healthy_dir,
                         target_pathologies, healthy_label, classes,
                         n_healthy, seed):
    """Reproduit fidèlement le pipeline de parsing/nettoyage/appariement/
    sous-échantillonnage de la Cellule 2 du baseline `2_0`.

    Renvoie un DataFrame avec les colonnes :
        filepath, File_Name, Last_Name, label_str
    """
    # ── Métadonnées FOCH ─────────────────────────────────────────────────────
    df_meta = pd.read_csv(csv_path)
    for col in ['Last_Name', 'File_Name', 'VSL']:
        df_meta[col] = df_meta[col].astype(str).str.strip()
    df_meta['VSL_norm'] = df_meta['VSL'].str.lower()

    # ── Filtrage sur les pathologies cibles ──────────────────────────────────
    target_map = {p.lower(): p for p in target_pathologies}
    df_patho = df_meta[df_meta['VSL_norm'].isin(target_map)].copy()
    df_patho['label_str'] = df_patho['VSL_norm'].map(target_map)

    # ── Déduplication patient ────────────────────────────────────────────────
    df_patho = df_patho.drop_duplicates(subset='Last_Name', keep='first')

    # ── Appariement robuste fichier CSV ↔ disque ─────────────────────────────
    def _resolve(name):
        p = os.path.join(foch_audio_dir, name)
        return p if os.path.isfile(p) else None
    df_patho['filepath'] = df_patho['File_Name'].apply(_resolve)
    df_patho = df_patho.dropna(subset=['filepath'])
    df_patho = df_patho[['filepath', 'File_Name', 'Last_Name', 'label_str']]

    # ── Groupe contrôle sain SVD (suffixe a_h), sous-échantillonné ───────────
    svd_files = glob.glob(os.path.join(svd_healthy_dir, '*', 'vowels', '*a_h.wav'))
    random.seed(seed)
    svd_files = random.sample(svd_files, min(n_healthy, len(svd_files)))
    df_healthy = pd.DataFrame({
        'filepath':  svd_files,
        'File_Name': [os.path.basename(f) for f in svd_files],
        'Last_Name': [os.path.basename(os.path.dirname(os.path.dirname(f))) for f in svd_files],
        'label_str': healthy_label,
    })

    df = pd.concat([df_patho, df_healthy], ignore_index=True)
    return df


def add_labels(df, classes):
    """Encode `label_str` → `label` (entier) et renvoie (df, label2id, id2label)."""
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}
    df = df.copy()
    df['label'] = df['label_str'].map(label2id).astype(int)
    assert df['label'].notna().all(), 'Libellé non mappé détecté.'
    return df, label2id, id2label


def split_train_val(df, seed, n_splits=5):
    """Découpage stratifié + groupé par patient (réplique de 2_0).
    Renvoie (train_df, val_df) — premier pli de StratifiedGroupKFold."""
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(df, df['label'], groups=df['Last_Name']))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    fuite = set(train_df['Last_Name']) & set(val_df['Last_Name'])
    assert not fuite, f'Fuite patient détectée : {sorted(fuite)}'
    return train_df, val_df


# ═══════════════════════════════════════════════════════════════════════════
# 2. Dataset PyTorch (identique à 2_0 ; lit une colonne `filepath`)
# ═══════════════════════════════════════════════════════════════════════════
class FochVowelDataset(Dataset):
    """Charge un enregistrement, mono + 16 kHz, puis applique l'extracteur de
    features WavLM. Renvoie (input_values, label)."""

    def __init__(self, dataframe, processor, target_sr=16000, max_len_s=4.0):
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor
        self.sr = target_sr
        self.max_len = int(target_sr * max_len_s)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row['label'])
        try:
            waveform, orig_sr = torchaudio.load(row['filepath'])
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if orig_sr != self.sr:
                waveform = torchaudio.transforms.Resample(orig_sr, self.sr)(waveform)
            wave_np = waveform.squeeze().numpy()
            out = self.processor(
                wave_np, sampling_rate=self.sr, max_length=self.max_len,
                truncation=True, padding='max_length', return_tensors='pt',
            )
            tensor = out.input_values.squeeze(0)
        except Exception as e:
            print(f'[ATTENTION] {row["filepath"]} : {e}')
            tensor = torch.zeros(self.max_len)
        return tensor, torch.tensor(label, dtype=torch.long)


def build_dataloaders(train_df, val_df, processor, batch_size,
                      target_sr=16000, max_len_s=4.0):
    train_ds = FochVowelDataset(train_df, processor, target_sr, max_len_s)
    val_ds = FochVowelDataset(val_df, processor, target_sr, max_len_s)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    return train_loader, val_loader


# ═══════════════════════════════════════════════════════════════════════════
# 3. Modèle WavLM-Large (avec contrôle explicite du masquage « SpecAugment »)
# ═══════════════════════════════════════════════════════════════════════════
def build_model(model_dir, num_labels, label2id, id2label, device,
                freeze_fe=True, spec_augment=False,
                mask_time_prob=0.08, mask_feature_prob=0.05):
    """Charge WavLM-Large pour la classification.

    `spec_augment` contrôle le masquage interne temps/fréquence de WavLM
    (équivalent SpecAugment pour les modèles à forme d'onde brute) : il n'est
    actif qu'en mode entraînement. On l'EXPOSE explicitement pour l'ablation —
    désactivé (False) pour le run signal-level pur, activé (True) sinon.
    """
    from transformers import WavLMForSequenceClassification
    model = WavLMForSequenceClassification.from_pretrained(
        model_dir, num_labels=num_labels, label2id=label2id, id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    # Contrôle explicite du masquage interne (SpecAugment).
    model.config.apply_spec_augment = bool(spec_augment)
    if spec_augment:
        model.config.mask_time_prob = mask_time_prob
        model.config.mask_feature_prob = mask_feature_prob
    else:
        model.config.mask_time_prob = 0.0
        model.config.mask_feature_prob = 0.0

    if freeze_fe:
        for p in model.wavlm.feature_extractor.parameters():
            p.requires_grad = False
    model.to(device)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 4. Boucle d'entraînement (AMP, accumulation, cosine, early stopping)
# ═══════════════════════════════════════════════════════════════════════════
def train_model(model, train_loader, val_loader, device, best_ckpt_path,
                epochs=30, lr=1e-5, weight_decay=0.01, accum_steps=4,
                patience=6, class_weights=None, verbose=True):
    """Entraîne le modèle, sauvegarde les meilleurs poids (perte de validation
    minimale). Renvoie l'historique {train_loss, val_loss, val_acc}."""
    from torch.amp import GradScaler, autocast
    from transformers import get_cosine_schedule_with_warmup

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=weight_decay)
    total_steps = max(1, len(train_loader) // accum_steps) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, total_steps // 10, total_steps)
    scaler = GradScaler()

    hist = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val, patience_ctr = float('inf'), 0

    for epoch in range(epochs):
        model.train()
        ep_tr, corr, tot = 0.0, 0, 0
        optimizer.zero_grad()
        for i, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with autocast(device_type=device.type):
                logits = model(xb).logits
                loss = criterion(logits, yb) / accum_steps
            scaler.scale(loss).backward()
            ep_tr += loss.item() * accum_steps
            corr += (logits.detach().argmax(1) == yb).sum().item()
            tot += yb.size(0)
            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        avg_tr = ep_tr / len(train_loader)

        model.eval()
        ep_v, corr_v, tot_v = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with autocast(device_type=device.type):
                    logits = model(xb).logits
                    loss = criterion(logits, yb)
                ep_v += loss.item()
                corr_v += (logits.argmax(1) == yb).sum().item()
                tot_v += yb.size(0)
        avg_v = ep_v / len(val_loader)
        hist['train_loss'].append(avg_tr)
        hist['val_loss'].append(avg_v)
        hist['val_acc'].append(corr_v / tot_v)

        if verbose:
            print(f'Epoch {epoch+1:02d}/{epochs} | '
                  f'Train {avg_tr:.4f}/{corr/tot:.3f} | Val {avg_v:.4f}/{corr_v/tot_v:.3f}')

        if avg_v < best_val:
            best_val = avg_v
            patience_ctr = 0
            torch.save(model.state_dict(), best_ckpt_path)
            if verbose:
                print(f'  → Meilleur modèle sauvegardé (Val Loss {best_val:.4f})')
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                if verbose:
                    print(f'Arrêt anticipé à l\'epoch {epoch+1}.')
                break
    return hist


# ═══════════════════════════════════════════════════════════════════════════
# 5. Évaluation clinique multi-classes (métriques robustes au déséquilibre)
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_model(model, val_loader, classes, device, history, metrics_dir,
                   run_name, best_ckpt_path=None, make_plots=True):
    """Recharge les meilleurs poids, évalue, sauvegarde dashboard + JSON + CSV.
    Renvoie le dictionnaire de métriques."""
    import torch.nn.functional as F
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_auc_score,
        balanced_accuracy_score, f1_score,
    )
    num_labels = len(classes)
    labels_idx = np.arange(num_labels)

    if best_ckpt_path and os.path.isfile(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb).logits
            y_prob.extend(F.softmax(logits, dim=1).cpu().numpy())
            y_pred.extend(logits.argmax(1).cpu().numpy())
            y_true.extend(yb.numpy())
    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)

    report = classification_report(y_true, y_pred, labels=labels_idx,
                                   target_names=classes, digits=3, zero_division=0)
    print('\n' + report)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    try:
        macro_auc = roc_auc_score(y_true, y_prob, multi_class='ovr',
                                  average='macro', labels=labels_idx)
    except ValueError:
        macro_auc = float('nan')
    print(f'[{run_name}] Balanced acc {bal_acc:.4f} | Macro-F1 {macro_f1:.4f} | '
          f'Macro AUC {macro_auc:.4f}')

    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

    if make_plots:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'FOCH — {run_name}\nBalanced Acc {bal_acc:.3f} | '
                     f'Macro-F1 {macro_f1:.3f} | Macro AUC {macro_auc:.3f}', fontsize=14)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                    xticklabels=classes, yticklabels=classes)
        axes[0, 0].set_title('Matrice de confusion (effectifs)')
        axes[0, 0].set_ylabel('Vraie classe'); axes[0, 0].set_xlabel('Classe prédite')
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', ax=axes[0, 1],
                    vmin=0, vmax=1, xticklabels=classes, yticklabels=classes)
        axes[0, 1].set_title('Matrice normalisée (rappel)')
        axes[0, 1].set_ylabel('Vraie classe'); axes[0, 1].set_xlabel('Classe prédite')
        f1_per = f1_score(y_true, y_pred, average=None, labels=labels_idx, zero_division=0)
        axes[1, 0].bar(classes, f1_per, color='steelblue')
        axes[1, 0].set_title('F1-score par classe'); axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=20)
        axes[1, 1].plot(range(1, len(history['train_loss']) + 1), history['train_loss'], 'o-', label='Train')
        axes[1, 1].plot(range(1, len(history['val_loss']) + 1), history['val_loss'], 'o-', label='Val')
        axes[1, 1].set_title('Courbes d\'apprentissage')
        axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Perte'); axes[1, 1].legend(); axes[1, 1].grid(True)
        plt.tight_layout(); plt.subplots_adjust(top=0.90)
        plt.savefig(os.path.join(metrics_dir, f'{run_name}_dashboard.png'), dpi=150, bbox_inches='tight')
        plt.show()

    report_dict = classification_report(y_true, y_pred, labels=labels_idx,
                                        target_names=classes, output_dict=True, zero_division=0)
    metrics = {
        'run_name': run_name,
        'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1,
        'macro_auc_ovr': macro_auc,
        'per_class': report_dict,
        'classes': classes,
    }
    with open(os.path.join(metrics_dir, f'{run_name}_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=float)
    np.savetxt(os.path.join(metrics_dir, f'{run_name}_confusion.csv'), cm, fmt='%d', delimiter=',')
    return metrics
