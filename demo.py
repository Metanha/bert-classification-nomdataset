import pandas as pd

import torch
import gradio as gr
from transformers import AutoTokenizer
from model import SentimentModel


data=pd.read_csv("data/Corona_NLP_train.csv", encoding="latin1")

data["Sentiment"]=data["Sentiment"].str.replace("Extremely ", "", regex=False)

<<<<<<< HEAD
import pandas as pd
import torch
import gradio as gr
from transformers import AutoTokenizer
from model import SentimentModel
=======
"""
MODEL_PATH         = "checkpoints/best_model.pth"
MODEL_NAME         = "google-bert/bert-base-uncased"
MAX_LENGTH         = 128
LABELS             = {0: "Neutral", 1: "Positive", 2: "Negative"}
>>>>>>> origin/feature-training


# ════════════════════════════════════════════
# Dataset — Exemples pré-remplis
# ════════════════════════════════════════════
data = pd.read_csv("data/Corona_NLP_train.csv", encoding="latin1")
data["Sentiment"] = data["Sentiment"].str.replace("Extremely ", "", regex=False)


# ════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════
MODEL_PATH = "best_model.pt"
MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 128
LABELS     = {0: "Positive", 1: "Negative", 2: "Neutral"}


# ════════════════════════════════════════════
# Device
# ════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")


# ════════════════════════════════════════════
# Tokenizer
# ════════════════════════════════════════════
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Tokenizer chargé")




model = SentimentModel(
    model_name_or_path = MODEL_NAME,
    n_class            = len(LABELS)
)


state_dict = torch.load(
    MODEL_PATH,
    map_location = device
)


model.load_state_dict(state_dict, strict=True)


model = model.to(device)
model.eval()
print(f"Modèle chargé depuis {MODEL_PATH}")



def predict(text):

    if not text.strip():
        return " Veuillez entrer un texte", {}

    ids = tokenizer(
        text,
        return_tensors = "pt",
        padding        = "max_length",
        truncation     = True,
        max_length     = MAX_LENGTH
    )

    input_ids      = ids["input_ids"].to(device)
    attention_mask = ids["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    probs      = torch.softmax(logits, dim=1)
    probs      = probs.squeeze(0).tolist()

    pred_idx   = torch.argmax(logits, dim=1).item()
    pred_label = LABELS[pred_idx]

    result = {LABELS[i]: round(probs[i], 4) for i in range(len(LABELS))}

    return pred_label, result


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""

    **Fine-tuning de BERT pour la classification de sentiments**
    Entrez un texte en anglais et le modèle prédit son sentiment :
    **Positif**, **Négatif** ou **Neutre**
    """)

    with gr.Row():

        with gr.Column():
            text_input = gr.Textbox(
                label       = "📝 Texte à analyser",
                placeholder = "Entrez votre texte ici...",
                lines       = 5
            )
            submit_btn = gr.Button(
                value   = "🔍 Analyser",
                variant = "primary"
            )

        with gr.Column():
            label_output = gr.Label(
                label = "Classe Prédite"
            )
            probs_output = gr.Label(
                label           = "📊 Probabilités par Classe",
                num_top_classes = 3
            )

    gr.Examples(
        examples = data.head()["OriginalTweet"].tolist(),
        inputs   = text_input,
        label    = "Exemples pré-remplis"
    )

    submit_btn.click(
        fn      = predict,
        inputs  = text_input,
        outputs = [label_output, probs_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
