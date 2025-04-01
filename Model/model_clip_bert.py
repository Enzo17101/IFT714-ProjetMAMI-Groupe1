
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, CLIPProcessor, CLIPModel
from PIL import Image
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


class MultimodalDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.data = dataframe
        self.image_dir = image_dir
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row["text"])
        file_name = row["file_name"]
        label = torch.tensor(row["label"], dtype=torch.long)

        # Texte
        text_inputs = self.bert_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

        # Image
        image_path = os.path.join(self.image_dir, file_name)
        image = Image.open(image_path).convert("RGB")
        image_inputs = self.clip_processor(images=image, return_tensors="pt")

        return {
            "text_input_ids": text_inputs["input_ids"].squeeze(0),
            "text_attention_mask": text_inputs["attention_mask"].squeeze(0),
            "image_pixel_values": image_inputs["pixel_values"].squeeze(0),
            "label": label
        }


class MultimodalClassifier(nn.Module):
    def __init__(self, bert_dim=768, clip_dim=512, hidden_dim=256):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.classifier = nn.Sequential(
            nn.Linear(bert_dim + clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, text_input_ids, text_attention_mask, image_pixel_values):
        text_outputs = self.bert(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_emb = text_outputs.pooler_output
        image_emb = self.clip.get_image_features(image_pixel_values)
        combined = torch.cat((text_emb, image_emb), dim=1)
        logits = self.classifier(combined)
        return logits


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            logits = model(
                text_input_ids=batch["text_input_ids"].to(device),
                text_attention_mask=batch["text_attention_mask"].to(device),
                image_pixel_values=batch["image_pixel_values"].to(device),
            )
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch["label"].numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    return acc, f1
