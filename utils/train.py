import torch


DEVICE = torch.device('cpu')


def entrainer(model, loader, criterion, optimizer):
    """Entraîne le modèle sur un epoch et retourne loss et accuracy."""
    model.train()
    loss_totale, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        loss_totale += loss.item()
        correct += (predictions.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return loss_totale / len(loader), correct / total


def evaluer(model, loader, criterion):
    """Évalue le modèle sans modifier les poids."""
    model.eval()
    loss_totale, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            predictions = model(images)
            loss = criterion(predictions, labels)

            loss_totale += loss.item()
            correct += (predictions.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return loss_totale / len(loader), correct / total


def entrainer_vae(model, loader, optimizer, epochs=30):
    """Entraîne le VAE et retourne l'historique des losses."""
    historique = {'total': [], 'recon': [], 'kl': []}

    for epoch in range(epochs):
        model.train()
        total_loss, total_recon, total_kl, n = 0, 0, 0, 0

        for images, _ in loader:
            optimizer.zero_grad()
            x_reconstruit, mu, log_var = model(images)
            loss, recon, kl = model.loss(images, x_reconstruit, mu, log_var)
            loss.backward()
            optimizer.step()

            total_loss  += loss.item() * images.size(0)
            total_recon += recon * images.size(0)
            total_kl    += kl * images.size(0)
            n += images.size(0)

        historique['total'].append(total_loss / n)
        historique['recon'].append(total_recon / n)
        historique['kl'].append(total_kl / n)

        if (epoch + 1) % 5 == 0:
            print(f"Époque {epoch+1:3d}/{epochs} | "
                  f"Total {historique['total'][-1]:.2f} | "
                  f"Recon {historique['recon'][-1]:.2f} | "
                  f"KL {historique['kl'][-1]:.2f}")

    return historique