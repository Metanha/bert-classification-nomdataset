import torch
import numpy as np
from torch.utils.data import Subset 
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm  # Suivi visuel de l'entraînement

from dataset import SentimentDataset
from model import SentimentModel

# Fonction pour calculer les métriques d'évaluation
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return acc, f1

# Fonction pour entraîner le modèle sur une époque
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    preds, true = [], []

    # Barre de progression interactive pour l'entraînement
    progress_bar = tqdm(loader, desc="Entraînement", leave=True)

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true.extend(labels.cpu().numpy())
        
        # Met à jour la perte en temps réel sur la ligne de terminal
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    acc, f1 = compute_metrics(true, preds)
    return total_loss / len(loader), acc, f1

# Fonction pour évaluer le modèle sur une époque
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, true = [], []

    # Barre de progression interactive pour la validation
    progress_bar = tqdm(loader, desc="Validation", leave=True)

    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true.extend(labels.cpu().numpy())

    acc, f1 = compute_metrics(true, preds)
    return total_loss / len(loader), acc, f1


# PROTECTION OBLIGATOIRE DU MULTIPROCESSING SOUS WINDOWS
if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Importation des données
    full_dataset = SentimentDataset(
        csv_file="data/Corona_NLP_train.csv",
        name_or_model_path="google-bert/bert-base-uncased"
    )

    # COMPROMIS SÉCURITÉ : Réduction à 10 000 lignes pour éviter la surchauffe thermique
    indices_reduits = np.arange(10000) 
    labels = full_dataset.data["Sentiment"].iloc[indices_reduits].values

    # Split du dataset stratifié
    train_idx, val_idx = train_test_split(
        indices_reduits,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # Création des sous-ensembles
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset   = Subset(full_dataset, val_idx)

    # Augmentation du batch_size à 16 pour diviser par deux le nombre d'itérations
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=0)

    # Récupération du nombre de classes
    n_classes = len(full_dataset.labels_dict)
    print("Classes:", n_classes)

    # Initialisation du modèle
    model = SentimentModel(
        model_name_or_path="google-bert/bert-base-uncased",
        n_class=n_classes
    )
    model.to(device)

    # Configuration de l'optimiseur et de la fonction de perte
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # Boucle globale d'entraînement (2 époques suffisent pour que BERT converge)
    epochs = 2
    best_val_loss = float("inf")

    print("\nLancement de l'entraînement sécurisé (10 000 lignes)...")
    for epoch in range(epochs):
        print(f"\n--- Époque {epoch+1} / {epochs} ---")
        
        # Phase d'entraînement
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Phase de validation
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, device)

        # Affichage du bilan de l'époque
        print(f"Train loss: {train_loss:.4f} | acc: {train_acc:.4f} | f1: {train_f1:.4f}")
        print(f"Val loss: {val_loss:.4f} | acc: {val_acc:.4f} | f1: {val_f1:.4f}")

        # Sauvegarde automatique du meilleur point de contrôle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print("=> MEILLEUR MODELE SAUVEGARDE (best_model.pt)")
