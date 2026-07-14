# Alzheimer Classification

Classification d'images IRM cérébrales pour détecter la maladie d'Alzheimer.

## Classes

- **Non_Demented** — Pas de démence
- **Very_Mild_Demented** — Démence très légère
- **Mild_Demented** — Démence légère

## Installation

```bash
pip install -r requirements.txt
```

## Téléchargement du dataset

```bash
python download_dataset.py
```

Après téléchargement, place les images dans la structure suivante :
```
dataset/
  train/
    Non_Demented/
    Very_Mild_Demented/
    Mild_Demented/
  val/
  test/
```

## Entraînement

```bash
python train.py
```

Le script affichera :
- Le nombre de classes
- Le nombre d'images dans chaque ensemble
- La loss à chaque epoch

## Modèle

CNN simple avec 3 couches de convolution :
- Conv1 → 32 filtres
- Conv2 → 64 filtres
- Conv3 → 128 filtres

## Configuration

| Paramètre | Valeur |
|-----------|--------|
| Taille image | 224x224 |
| Batch size | 32 |
| Epochs | 10 |
| Optimizer | Adam (lr=0.001) |
| Loss | CrossEntropyLoss |
