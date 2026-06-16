# Import de PyTorch
import torch

# Import du module de base pour créer des réseaux de neurones
import torch.nn as nn

# Import d'un modèle pré-entraîné générique depuis Hugging Face
from transformers import AutoModel


# Définition d'un modèle de classification de sentiment
class SentimentModel(nn.Module):

    def __init__(self, model_name_or_path, n_class):
        """
        Constructeur du modèle.

        Paramètres :
        - model_name_or_path : nom du modèle pré-entraîné
          ex: 'bert-base-uncased'
        - n_class : nombre de classes à prédire
          ex: 2 pour positif/négatif, 3 pour positif/neutre/négatif
        """
        super().__init__()

        # Chargement du modèle Transformer pré-entraîné
        # Exemple : BERT, CamemBERT, DistilBERT, etc.
        self.pretrained = AutoModel.from_pretrained(model_name_or_path)

        # Récupération de la taille des vecteurs de sortie du modèle
        # hidden_size correspond à la dimension de la représentation d'un token
        self.dim_in = self.pretrained.config.hidden_size  

        # Couche linéaire finale pour faire la classification
        # Elle transforme le vecteur CLS en logits de taille n_class
        self.proj_lin = nn.Linear(self.dim_in, n_class)

    def forward(self, input_ids, attention_mask):
        """
        Passage avant du modèle.

        Paramètres :
        - input_ids : identifiants numériques des tokens
        - attention_mask : masque indiquant quels tokens sont réels
          et quels tokens sont du padding

        Retour :
        - logits : scores bruts pour chaque classe
        """

        # Passage des données dans le modèle pré-entraîné
        outputs = self.pretrained(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # On récupère la représentation du token [CLS]
        # last_hidden_state a la forme :
        # (batch_size, sequence_length, hidden_size)
        #
        # [:, 0, :] signifie :
        # - toutes les phrases du batch
        # - le premier token de chaque phrase (souvent [CLS])
        # - toutes les dimensions du vecteur
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Passage du vecteur CLS dans la couche linéaire
        # pour obtenir les scores de classification
        logits = self.proj_lin(cls_output)

        # Retour des scores bruts
        return logits
