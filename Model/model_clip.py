import pandas as pd
import os
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm  # Importer tqdm pour la barre de progression

# Charger les données
train_df = pd.read_csv("../preprocessing_for_CLIP/train.csv")
test_texts_df = pd.read_csv("../preprocessing_for_CLIP/test_texts.csv")
test_labels_df = pd.read_csv("../preprocessing_for_CLIP/test_labels.csv")
train_folder_path = "../dataset/TRAINING"
test_folder_path = "../dataset/test"


# Compter le nombre d'exemples dans chaque classe
print("Répartition des labels dans le dataset d'entraînement :\n", train_df["label"].value_counts())

# Charger le modèle CLIP et le processor pour prétraiter les images et textes
print("Chargement du modèle CLIP...")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")  # Retirer use_fast=True
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
print("Modèle CLIP chargé !")

def image_path_to_tensor(image_path, folder_path):
    # Charger l'image avec PIL
    full_image_path = os.path.join(folder_path, image_path)
    image = Image.open(full_image_path).convert("RGB")
    image_np = np.array(image)/255.0  # Normaliser l'image
    # Vérifier si les valeurs sont bien dans la plage [0, 1]
    if np.any(image_np < 0) or np.any(image_np > 1):
        print(f"Attention: image avec valeurs hors de la plage [0, 1] dans {image_path}")
    # Utiliser le préprocesseur CLIP pour transformer l'image
    inputs = processor(images=image, return_tensors="pt", padding=True, do_rescale=False)  # Désactiver la mise à l'échelle

    return inputs['pixel_values'][0]  # Retourne le tensor de l'image

# Fonction pour obtenir les embeddings à partir de CLIP
def get_clip_embeddings(texts, images):
    # Prétraiter les textes séparément
    #on affiche les dimensions des images
    print(images[0].shape)
    text_inputs = processor(text=texts, return_tensors="pt", padding=True)
    # Prétraiter les images séparément
    images_normalized = [image / 255.0 if isinstance(image, np.ndarray) else np.array(image) / 255.0 for image in images]

    image_inputs = processor(images=images_normalized, return_tensors="pt", padding=True, do_rescale=False)  # Désactiver la mise à l'échelle

    with torch.no_grad():
        # Obtenir les embeddings des textes et des images
        text_embeds = model.get_text_features(**text_inputs)
        image_embeds = model.get_image_features(**image_inputs)

    return text_embeds, image_embeds

# Dataset pour charger les textes et les images
class MemeDataset(Dataset):
    def __init__(self, texts, images, labels):
        self.texts = texts
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        image = self.images[idx]  # Assurez-vous que les images sont chargées sous forme de tensor ou de chemin vers l'image
        label = self.labels[idx]
        return text, image, label

# Charger les images (supposons qu'elles sont en format chemin d'image ou déjà converties en tensors)
print("Chargement des images de train")
train_images = [image_path_to_tensor(image_path, train_folder_path) for image_path in train_df["file_name"]]
print("Chargement des images de test")
test_images = [image_path_to_tensor(image_path, test_folder_path) for image_path in test_texts_df["file_name"]]

# Préparer les datasets
print("Préparation des datasets...")
train_texts = train_df["text"].tolist()
train_labels = train_df["label"].tolist()

test_texts = test_texts_df["text"].tolist()
test_labels = test_labels_df["label"].tolist()

train_dataset = MemeDataset(train_texts, train_images, train_labels)
test_dataset = MemeDataset(test_texts, test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)
print("Préparation des datasets terminée !")

print("Entraînement du modèle de régression logistique...")
# Entraîner un modèle de régression logistique sur les embeddings CLIP
log_reg = LogisticRegression()
print("Entraînement du modèle de régression logistique terminé !")

# Extraire les embeddings et entraîner le modèle
def extract_embeddings(loader):
    embeddings = []
    labels = []
    # Ajouter tqdm pour la barre de progression
    for texts, images, label in tqdm(loader, desc="Extraction des embeddings", leave=False):
        text_embeds, image_embeds = get_clip_embeddings(texts, images)
        combined_embeds = torch.cat((text_embeds, image_embeds), dim=-1).numpy()  # Combine embeddings
        embeddings.append(combined_embeds)
        labels.append(label.numpy())
    return np.vstack(embeddings), np.hstack(labels)

# Extraire les embeddings d'entraînement et de test
print("Extraction des embeddings d'entraînement...")
X_train, y_train = extract_embeddings(train_loader)
X_test, y_test = extract_embeddings(test_loader)

print("Extraction des embeddings terminée !")

# Entraîner le modèle
log_reg.fit(X_train, y_train)

# Prédictions
y_pred_train = log_reg.predict(X_train)
y_pred_test = log_reg.predict(X_test)

# Calculer les métriques
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

train_f1 = f1_score(y_train, y_pred_train, average='binary')
test_f1 = f1_score(y_test, y_pred_test, average='binary')

print(f"Train Accuracy: {train_acc}")
print(f"Test Accuracy: {test_acc}")
print(f"Train F1 Score: {train_f1}")
print(f"Test F1 Score: {test_f1}")
