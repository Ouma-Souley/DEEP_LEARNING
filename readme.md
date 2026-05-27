# Projet Deep Learning — Classification d'images sur CIFAR-10

**Auteur** : Oumalheir Souley Na Lado  
**Cours** : Deep Learning: Models and Optimization — ENSAE Paris  
**Date** : Mai 2026

## Problématique

Comment améliorer la généralisation d'un modèle de classification d'images
en combinant apprentissage discriminatif et génératif ?

## Structure du projet
Deep_Learning/
├── projet.ipynb          ← Notebook principal
├── models/
│   ├── mlp.py            ← MLP (Perceptron Multicouche)
│   ├── cnn.py            ← CNN (Réseau Convolutif)
│   └── vae.py            ← VAE (Variational Autoencoder)
├── utils/
│   ├── train.py          ← Fonctions d'entraînement et d'évaluation
│   └── visualize.py      ← Fonctions de visualisation
├── fig_*.png             ← Figures générées
└── README.md

## Installation

```bash
pip install torch torchvision numpy matplotlib scikit-learn seaborn tqdm pandas jupyter
```

## Lancement

Ouvrir `projet.ipynb` dans VSCode et exécuter les cellules dans l'ordre.

## Résultats

| Modèle | Approche | Acc. test |
|--------|----------|-----------|
| MLP plain | Discriminatif | 53.4% |
| MLP dropout | Discriminatif | 51.6% |
| CNN plain | Discriminatif | 81.4% |
| CNN augmenté | Discriminatif | 83.3% |
| VAE + LogReg | Génératif → Discr. | 43.1% |

## Figures produites

| Figure | Description |
|--------|-------------|
| fig_dataset.png | Exemples CIFAR-10 |
| fig_mlp_plain.png | Surapprentissage du MLP |
| fig_mlp_comparaison.png | MLP plain vs dropout |
| fig_mlp_vs_cnn.png | MLP vs CNN |
| fig_cnn_augmentation.png | Impact data augmentation |
| fig_robustesse.png | Robustesse au bruit gaussien |
| fig_vae_loss.png | Courbes d'entraînement VAE |
| fig_vae_reconstructions.png | Originaux vs reconstructions |
| fig_tsne_vae.png | Espace latent t-SNE |
| fig_interpolation.png | Interpolation avion → chien |

## Conclusion

Le CNN augmenté est le meilleur modèle discriminatif (83.3%).
Le VAE démontre de bonnes capacités génératives mais ses features
non supervisées restent moins discriminantes que celles du CNN.