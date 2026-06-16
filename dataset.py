import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class SentimentDataset(Dataset):

    def __init__(self, csv_file, name_or_model_path):

        # Chargement du tokenizer à partir du modèle pré-entraîné
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_model_path)

        # Lecture du fichier CSV contenant les tweets et leurs sentiments
        self.data = pd.read_csv(csv_file, encoding="latin-1")
        self.data = self.data.copy()

        # Simplification des labels en supprimant le préfixe "Extremely"
        # par exemple "Extremely Positive" devient "Positive"
        self.data["Sentiment"] = self.data["Sentiment"].str.replace("Extremely ", "", regex=False)

        # On garde uniquement les colonnes utiles
        self.data = self.data[["OriginalTweet", "Sentiment"]]

        # Création d'un dictionnaire qui associe chaque label à un entier
        # par exemple {"Positive": 0, "Negative": 1, "Neutral": 2}
        self.labels_dict = dict()
        for indx, l in enumerate(self.data["Sentiment"].unique()):
            self.labels_dict[l] = indx

    def __len__(self):
        # Retourne le nombre total d'exemples dans le dataset
        return len(self.data)

    def __getitem__(self, idx):

        # Récupération du tweet et de son sentiment à l'index idx
        text  = self.data["OriginalTweet"][idx]
        label = self.data["Sentiment"][idx]

        # Tokenisation du texte avec padding et troncature
        # pour que tous les textes aient la même longueur de 200 tokens
        ids = self.tokenizer(
            text,
            return_tensors = "pt",
            padding        = "max_length",
            truncation     = True,
            max_length     = 200
        )

        # Conversion du label textuel en entier via le dictionnaire
        label = self.labels_dict[label]

        # Retourne un dictionnaire contenant les tenseurs nécessaires
        # pour l'entraînement du modèle BERT
        return {
            # Les identifiants des tokens, squeeze pour supprimer la dimension batch
            "input_ids"      : ids["input_ids"].squeeze(0),
            # Le masque d'attention pour ignorer les tokens de padding
            "attention_mask" : ids["attention_mask"].squeeze(0),
            # Le label converti en tenseur
            "label"          : torch.tensor(label)
        }
        

if __name__ == "__main__":

    # Création du dataset à partir du fichier CSV et du modèle BERT
    dataset = SentimentDataset(
        csv_file           = "data/Corona_NLP_train.csv",
        name_or_model_path = "google-bert/bert-base-uncased"
    )

    # Création du dataloader avec un batch de 2 exemples
    dataloader = DataLoader(dataset=dataset, batch_size=2)

    # Récupération et affichage du premier batch pour vérification
    data = next(iter(dataloader))
    print(data)
