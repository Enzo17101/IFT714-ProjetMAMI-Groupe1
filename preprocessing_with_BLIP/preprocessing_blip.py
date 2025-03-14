import pandas as pd
import re
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Charger le modèle et le processor BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Fonction pour générer la description de l'image avec BLIP
def generate_image_description(image_path):
    print(f"Génération de la description pour l'image {image_path}...")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    # Générer la description
    out = model.generate(**inputs)

    return processor.decode(out[0], skip_special_tokens=True)

# Nettoyage du texte (minuscules, suppression des caractères spéciaux)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()


# NETTOYAGE DU FICHIER TRAIN

# Chemin du fichier
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "training"))
file_path = os.path.join(base_path, "training.csv")

df = pd.read_csv(file_path, sep="\t")

# On garde la colonne qui précise si le mème est misogyne ou non, le texte du mème et le nom du fichier
df = df[['misogynous', 'Text Transcription', 'file_name']]

# Appliquer le nettoyage et renommer les colonnes
df['Text Transcription'] = df['Text Transcription'].apply(clean_text)
df = df.rename(columns={'misogynous': 'label', 'Text Transcription': 'text'})

# Génération des descriptions d'images
df['img_desc'] = df['file_name'].apply(lambda x: generate_image_description(os.path.join(base_path, x)))
df.drop(columns=['file_name'], inplace=True)

# Sauvegarder le nouveau CSV
df.to_csv("train_blip.csv", index=False)
print("Fichier train_blip.csv créé avec succès !")


# NETTOYAGE DU FICHIER TEST

# Chemin des fichiers
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "test"))
file_path_test = os.path.join(base_path, "Test.csv")
file_path_labels = os.path.join(base_path, "test_labels.txt")

df_test = pd.read_csv(file_path_test, sep="\t")
df_labels = pd.read_csv(file_path_labels, sep="\t", header=None, names=["file_name", "misogynous", "shaming", "stereotype", "objectification", "violence"])

# Fusionner les deux DataFrames sur la colonne 'file_name'
df = pd.merge(df_test, df_labels, on="file_name")

# On garde la colonne qui précise si le mème est misogyne ou non, le texte du mème et le nom du fichier
df = df[['misogynous', 'Text Transcription', 'file_name']]

# Appliquer le nettoyage et renommer les colonnes
df['Text Transcription'] = df['Text Transcription'].apply(clean_text)
df = df.rename(columns={'misogynous': 'label', 'Text Transcription': 'text'})

# Génération des descriptions d'images
df['img_desc'] = df['file_name'].apply(lambda x: generate_image_description(os.path.join(base_path, x)))
df.drop(columns=['file_name'], inplace=True)

# Sauvegarder le nouveau CSV
df.to_csv("test_blip.csv", index=False)
print("Fichier test_blip.csv créé avec succès !")
