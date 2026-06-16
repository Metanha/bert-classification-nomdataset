import pandas as pd

import torch
import gradio as gr
from transformers import AutoTokenizer
from model import SentimentModel

data=pd.read_csv("data/Corona_NLP_train.csv", encoding="latin1")

data["Sentiment"]=data["Sentiment"].str.replace("Extremely ", "", regex=False)

"""
MODEL_PATH         = "checkpoints/best_model.pth"
MODEL_NAME         = "google-bert/bert-base-uncased"
MAX_LENGTH         = 128
LABELS             = {0: "Positive", 1: "Negative", 2: "Neutral"}


# ════════════════════════════════════════════
# Chargement Modèle + Tokenizer
# ════════════════════════════════════════════
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model     = torch.load(MODEL_PATH, map_location=device)
model.eval()


# ════════════════════════════════════════════
# Fonction de Prédiction
# ════════════════════════════════════════════
def predict(text):

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
"""

# Interface Gradio

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    # ── Titre + Description
    gr.Markdown("""
    #Sentiment Analysis avec BERT
    **Fine-tuning de BERT pour la classification de sentiments**
    Entrez un texte en anglais et le modèle prédit son sentiment :
    **Positif**, **Négatif** ou **Neutre**
    """)

    with gr.Row():

        # ── Colonne Gauche : Input
        with gr.Column():
            text_input = gr.Textbox(
                label       = "Texte à analyser",
                placeholder = "Entrez votre texte ici...",
                lines       = 5
            )
            submit_btn = gr.Button(
                value   = "Analyser",
                variant = "primary"
            )

        # ── Colonne Droite : Output
        with gr.Column():
            label_output = gr.Label(
                label = "Classe Prédite"
            )
            probs_output = gr.Label(
                label     = "Probabilités par Classe",
                num_top_classes = 3
            )

    # ── Exemples pré-remplis
    gr.Examples(
        examples = data.head()["OriginalTweet"].tolist(),
        inputs  = text_input,
        label   = "Exemples pré-remplis"
        
       
        

    )

    # ── Action du bouton
    """
    submit_btn.click(
        fn      = predict,
        inputs  = text_input,
        outputs = [label_output, probs_output]
    )
"""

# ════════════════════════════════════════════
# LANCEMENT
# ════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch(share=True)
