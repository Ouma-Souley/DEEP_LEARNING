import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

CLASSES = ('avion', 'auto', 'oiseau', 'chat', 'cerf',
           'chien', 'grenouille', 'cheval', 'bateau', 'camion')


def plot_histories(histories, title, filename):
    """Trace les courbes accuracy pour plusieurs modèles."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, h in histories.items():
        ax.plot(h['train_acc'], linestyle='--', label=f'{name} — train')
        ax.plot(h['test_acc'],               label=f'{name} — test')
    ax.set(title=title, xlabel='Époque', ylabel='Accuracy')
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_robustesse(sigmas, accs, labels, filename):
    """Trace les courbes de robustesse au bruit."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for acc, label in zip(accs, labels):
        ax.plot(sigmas, acc, 'o-', label=label)
    ax.axhline(y=0.1, color='red', linestyle='--', label='Hasard (10%)')
    ax.set(title='Robustesse au bruit gaussien',
           xlabel='σ', ylabel='Accuracy test')
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_reconstructions(model, loader, filename):
    """Affiche originaux vs reconstructions du VAE."""
    def denorm(t):
        return ((t * 0.5) + 0.5).clamp(0, 1)

    model.eval()
    images, _ = next(iter(loader))
    images = images[:10]
    with torch.no_grad():
        reconstructions, _, _ = model(images)

    fig, axes = plt.subplots(2, 10, figsize=(16, 4))
    for i in range(10):
        axes[0, i].imshow(denorm(images[i]).permute(1,2,0).numpy())
        axes[0, i].axis('off')
        axes[1, i].imshow(denorm(reconstructions[i]).permute(1,2,0).numpy())
        axes[1, i].axis('off')
    plt.suptitle('VAE — Originaux vs Reconstructions', fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_tsne(model, loader, filename, n=2000):
    """Visualise l'espace latent du VAE avec t-SNE."""
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for images, labels in loader:
            zs.append(model.encode(images))
            ys.append(labels)
            if sum(len(z) for z in zs) >= n:
                break

    Z = torch.cat(zs)[:n].numpy()
    Y = torch.cat(ys)[:n].numpy()

    Z_2d = TSNE(n_components=2, perplexity=40,
                random_state=42, max_iter=1000).fit_transform(Z)

    couleurs = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_idx, (cls_nom, couleur) in enumerate(zip(CLASSES, couleurs)):
        masque = Y == cls_idx
        ax.scatter(Z_2d[masque, 0], Z_2d[masque, 1],
                   c=[couleur], label=cls_nom, alpha=0.6, s=12)
    ax.set(title='Espace latent du VAE — t-SNE',
           xlabel='Composante 1', ylabel='Composante 2')
    ax.legend(markerscale=2, fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()