import torch
import torch.nn as nn
from transformers import AutoModel


class SentimentModel(nn.Module):

    def __init__(self, model_name_or_path, n_class):
        super().__init__()

   
        self.pretrained = AutoModel.from_pretrained(model_name_or_path)

        self.dim_in = self.pretrained.config.hidden_size  

        self.proj_lin = nn.Linear(self.dim_in, n_class)

    def forward(self, input_ids, attention_mask):

 
        outputs = self.pretrained(
            input_ids      = input_ids
        )

        cls_output = outputs.last_hidden_state[:, 0, :]

        logits = self.proj_lin(cls_output)

        return logits

