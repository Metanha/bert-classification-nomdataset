import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class SentimentDataset(Dataset):

    def __init__(self, csv_file, name_or_model_path):

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_model_path)

        self.data = pd.read_csv(csv_file, encoding="latin-1")
        self.data = data.copy()
        self.data["Sentiment"] = self.data["Sentiment"].str.replace("Extremely ", "", regex=False)
        self.data = self.data[["OriginalTweet", "sentiment"]]

        self.labels_dict = dict()
        for indx, l in enumerate(self.data.sentiment.unique()):
            self.labels_dict[l] = indx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        text  = self.data["text"][idx]
        label = self.data["sentiment"][idx]

        
        ids = self.tokenizer(
            text,
            return_tensors = "pt",
            padding        = "max_length",
            truncation     = True,
            max_length     = 200
        )

        label = self.labels_dict[label]

        return {
            "input_ids"      : ids["input_ids"].squeeze(0),      
            "attention_mask" : ids["attention_mask"].squeeze(0), 
            "label"          : torch.tensor(label)
        }
        
    

if __name__ == "__main__":

    dataset = SentimentDataset(
        csv_file           = "train.csv",
        name_or_model_path = "google-bert/bert-base-uncased"
    )

    dataloader = DataLoader(dataset=dataset, batch_size=2)

    data = next(iter(dataloader))
    print(data)

