import pandas as pd
import torch
import gradio as gr
from transformers import AutoTokenizer
from model import SentimentModel


# Chargement du dataset et simplification des labels
# "Extremely Positive" devient "Positive" par exemple
data = pd.read_csv("data/Corona_NLP_train.csv", encoding="latin1")
data["Sentiment"] = data["Sentiment"].str.replace("Extremely ", "", regex=False)


# Paramètres généraux de l'application
MODEL_PATH = "best_model.pt"
MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 128
LABELS     = {0: "Positive", 1: "Negative", 2: "Neutral"}


# Utilisation du GPU si disponible, sinon on utilise le CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")


# Chargement du tokenizer BERT
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Tokenizer chargé")


# Chargement du modèle en trois étapes :
# on recrée l'architecture, on charge les poids sauvegardés, puis on les injecte
model = SentimentModel(
    model_name_or_path = MODEL_NAME,
    n_class            = len(LABELS)
)

state_dict = torch.load(
    MODEL_PATH,
    map_location = device
)

model.load_state_dict(state_dict, strict=True)

# On envoie le modèle sur le bon device et on le passe en mode évaluation
# le mode évaluation désactive le dropout et la normalisation par batch
model = model.to(device)
model.eval()
print(f"Modèle chargé depuis {MODEL_PATH}")


# Fonction de prédiction appelée à chaque soumission de texte
def predict(text):

    # Si le texte est vide on retourne un message d'avertissement
    if not text.strip():
        return "Veuillez entrer un texte", {}

    # Tokenisation du texte avec padding et troncature à 128 tokens
    ids = tokenizer(
        text,
        return_tensors = "pt",
        padding        = "max_length",
        truncation     = True,
        max_length     = MAX_LENGTH
    )

    # Envoi des tenseurs sur le bon device
    input_ids      = ids["input_ids"].to(device)
    attention_mask = ids["attention_mask"].to(device)

    # Inférence sans calcul de gradient pour économiser la mémoire
    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    # Conversion des logits en probabilités avec softmax
    probs      = torch.softmax(logits, dim=1)
    probs      = probs.squeeze(0).tolist()

    # Récupération du label avec la probabilité la plus haute
    pred_idx   = torch.argmax(logits, dim=1).item()
    pred_label = LABELS[pred_idx]

    # Construction du dictionnaire de résultats label -> probabilité
    result = {LABELS[i]: round(probs[i], 4) for i in range(len(LABELS))}

    return pred_label, result


# Construction de l'interface Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # Sentiment Analysis avec BERT
    **Fine-tuning de BERT pour la classification de sentiments**
    Entrez un texte en anglais et le modèle prédit son sentiment :
    **Positif**, **Negatif** ou **Neutre**
    """)

    with gr.Row():

        with gr.Column():
            # Zone de saisie du texte à analyser
            text_input = gr.Textbox(
                label       = "Texte à analyser",
                placeholder = "Entrez votre texte ici...",
                lines       = 5
            )
            submit_btn = gr.Button(
                value   = "Analyser",
                variant = "primary"
            )

        with gr.Column():
            # Affichage du label prédit
            label_output = gr.Label(
                label = "Classe Predite"
            )
            # Affichage des probabilités pour chaque classe
            probs_output = gr.Label(
                label           = "Probabilites par Classe",
                num_top_classes = 3
            )

    # Exemples pre-remplis tirés des premiers tweets du dataset
    gr.Examples(
        examples = data.head()["OriginalTweet"].tolist(),
        inputs   = text_input,
        label    = "Exemples pre-remplis"
    )

    # Connexion du bouton à la fonction de prédiction
    submit_btn.click(
        fn      = predict,
        inputs  = text_input,
        outputs = [label_output, probs_output]
    )


# Lancement de l'application avec un lien public partageable
if __name__ == "__main__":
    demo.launch(share=True)
