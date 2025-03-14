#!/usr/bin/env python
# coding: utf-8

# In[116]:


import pandas as pd
import torch
from pywin32_testutil import testmain
from sklearn.model_selection import train_test_split, learning_curve
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from transformers import default_data_collator
#from transformers.agents.evaluate_agent import classifier
import numpy as np
import evaluate
from sklearn.metrics import f1_score


# ### Hyperparamètres

# In[117]:


# Tokeniser
max_length = 128 # Longueur maximale des séquences tokenisées

# Capacité du modèle
batch_size = 16 # Taille des batchs
num_epochs = 3 # Nombre d'époques

# Dropout
dropout_global = 0.4 # Dropout global
dropout_attention = 0.4 # Dropout dans les couches d'attention

# Autres paramètres du modèle
#learning_rate = 2e-5 # Taux d'apprentissage
weight_decay = 0.1 # Paramètre de régularisation L2
warmup_steps = 0 # Pas de warmup
load_best_model_at_end = True # Charger le meilleur modèle parmi ceux générés pendant les epochs

# Nombre de labels dans le dataset
num_labels = 2


# In[118]:


# Détection du matériel à disposition pour l'entrainement du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cpu= True if device == "cpu" else False

if str(device) == "cuda":
    device_name = torch.cuda.get_device_name()
else:
    device_name = "CPU nul..."

print("Device:", device_name)


# ## Import des données

# In[119]:


# Charger les données
train_df = pd.read_csv("../preprocessing/train.csv")
test_df = pd.read_csv("../preprocessing/test.csv")

# Nettoyage des données
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# Compter le nombre d'exemples dans chaque classe dans le dataset d'entraînement
train_label_counts = train_df["label"].value_counts()
print("Répartition des labels dans le dataset d'entraînement :\n", train_label_counts)
print()

# Compter le nombre d'exemples dans chaque classe dans le dataset de test
test_label_counts = test_df["label"].value_counts()
print("Répartition des labels dans le dataset de test :\n", test_label_counts)


# In[120]:


# Tokenizer de DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


# In[121]:


class MemeDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# ## Préparation et création du modèle

# In[122]:


from transformers import DistilBertConfig

# Séparer 10-20% des données d'entraînement pour la validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df["text"], train_df["label"], test_size=0.1, random_state=42
)

# Créer les datasets pour Trainer
train_dataset = MemeDataset(train_texts.tolist(), train_labels.tolist())
val_dataset = MemeDataset(val_texts.tolist(), val_labels.tolist())
test_dataset = MemeDataset(test_df["text"].tolist(), test_df["label"].tolist())

config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
config.dropout = dropout_global  # Appliquer du dropout globalement
config.attention_dropout = dropout_attention  # Appliquer du dropout dans les couches d'attention
config.num_labels = num_labels  # Nombre de labels dans le dataset

# Charger le modèle DistilBERT (sur le device détecté)
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", config=config)
_ = model.to(device)


# In[123]:


# Configuration de l'entraînement optimisée pour CPU
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=num_epochs,  # Réduire à 1 époque pour éviter un entraînement trop long
    per_device_train_batch_size=batch_size,  # Réduire pour éviter saturation mémoire
    per_device_eval_batch_size=batch_size,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    lr_scheduler_type="cosine",
    logging_dir="./logs",
    eval_strategy="epoch",
    save_strategy="epoch",
    use_cpu=use_cpu,
    load_best_model_at_end = True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator  # Ajout de cette ligne
)


# ## Entrainement du modèle

# In[124]:


# Entraînement du modèle (cela prendra du temps sur CPU)
print(f"Modèle chargé sur {device_name}")
print("Début de l'entrainement...")
trainer.train()


# In[125]:


# Obtenir les prédictions sur le train set
predictions = trainer.predict(train_dataset)
logits = predictions.predictions
y_train = np.argmax(logits, axis=-1)

# Obtenir les prédictions sur le test set
predictions = trainer.predict(test_dataset)
logits = predictions.predictions
y_pred = np.argmax(logits, axis=-1)

# Calculer la distribution des prédictions pour le train et le test
unique_train, counts_train = np.unique(y_train, return_counts=True)
train_counts = dict(zip(unique_train, counts_train))

unique_test, counts_test = np.unique(y_pred, return_counts=True)
test_counts = dict(zip(unique_test, counts_test))

# Afficher la distribution avec les pourcentages
print("=== Distribution des prédictions sur TRAIN ===")
total_train = sum(counts_train)
for label, count in train_counts.items():
    percentage = (count / total_train) * 100
    print(f"Classe {label}: {count} ({percentage:.2f}%)")
print("=================================")

print("=== Distribution des prédictions sur TEST ===")
total_test = sum(counts_test)
for label, count in test_counts.items():
    percentage = (count / total_test) * 100
    print(f"Classe {label}: {count} ({percentage:.2f}%)")
print("=================================")

# Charger la métrique d'accuracy
accuracy_metric = evaluate.load("accuracy")

# Obtenir les prédictions sur le train et dev set
train_predictions = trainer.predict(train_dataset)
train_logits = train_predictions.predictions
train_labels = train_predictions.label_ids
train_preds = np.argmax(train_logits, axis=-1)

test_predictions = trainer.predict(test_dataset)
test_logits = test_predictions.predictions
test_labels = test_predictions.label_ids
test_preds = np.argmax(test_logits, axis=-1)

# Calculer l'accuracy
train_acc = accuracy_metric.compute(predictions=train_preds, references=train_labels)['accuracy']
test_acc = accuracy_metric.compute(predictions=test_preds, references=test_labels)['accuracy']

# Calculer le F1 score
train_f1 = f1_score(train_labels, train_preds, average='binary')
test_f1 = f1_score(test_labels, test_preds, average='binary')

# Distribution des prédictions en pourcentage
def distribution(y_pred):
    unique, counts = np.unique(y_pred, return_counts=True)
    total = len(y_pred)
    distribution = {label: round((count / total) * 100, 2) for label, count in zip(unique, counts)}
    return distribution

train_dist = distribution(train_preds)
dev_dist = distribution(test_preds)

# Affichage des résultats
print("===== Train Accuracy =====")
acc = round(train_acc * 100, 2)
print(f"Accuracy: {train_acc}%")

acc = round(train_f1 * 100, 2)
print(f"F1 Score: {acc}%")

print("\n===== Dev Accuracy =====")
acc = round(test_acc * 100, 2)
print(f"Accuracy: {acc}%")

acc = round(test_f1 * 100, 2)
print(f"F1 Score: {acc}%")


# ## Export des résultats et hyperparamètres associés

# In[126]:


import csv

# Hyperparamètres et résultats
params = {
    "max_length": max_length,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "dropout_global": dropout_global,
    "dropout_attention": dropout_attention,
    "weight_decay": weight_decay,
    "warmup_steps": warmup_steps,
    "load_best_model_at_end": load_best_model_at_end,
    "num_labels": num_labels,
    "train_acc": train_acc,
    "test_acc": test_acc,
    "train_pred_distribution": train_counts,
    "test_pred_distribution": test_counts
}

# Nom du fichier CSV
output_file = "hyperparameters_results.csv"

# Écriture dans le CSV
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Écriture des hyperparamètres
    writer.writerow(["Hyperparamètres et Résultats", "Valeurs"])
    for key, value in params.items():
        if isinstance(value, dict):
            value = str(value)  # Convertir les distributions en string pour CSV
        writer.writerow([key, value])

print(f"Données exportées avec succès dans {output_file}!")

