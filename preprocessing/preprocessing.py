import pandas as pd
import re
import os

# NETTOYAGE DU FICHIER TRAIN

# Chemin du fichier
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "training"))
file_path = os.path.join(base_path, "training.csv")

df = pd.read_csv(file_path, sep="\t")

# On garde la colonne qui précise si le mème est misogyne ou non et le texte du mème
df = df[['misogynous', 'Text Transcription']]

# Nettoyage du texte (minuscules, suppression des caractères spéciaux)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

# Appliquer le nettoyage et renommer les colonnes
df['Text Transcription'] = df['Text Transcription'].apply(clean_text)
df = df.rename(columns={'misogynous': 'label', 'Text Transcription': 'text'})

# Sauvegarder le nouveau CSV
df.to_csv("train.csv", index=False)
print("Fichier train.csv créé avec succès !")


# NETTOYAGE DU FICHIER TEST

# Chemin des fichiers
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "test"))
file_path_test = os.path.join(base_path, "Test.csv")
file_path_labels = os.path.join(base_path, "test_labels.txt")

df_test = pd.read_csv(file_path_test, sep="\t")
df_labels = pd.read_csv(file_path_labels, sep="\t", header=None, names=["file_name", "misogynous", "shaming", "stereotype", "objectification", "violence"])

# Fusionner les deux DataFrames sur la colonne 'file_name'
df = pd.merge(df_test, df_labels, on="file_name")

# On garde la colonne qui précise si le mème est misogyne ou non et le texte du mème
df = df[['misogynous', 'Text Transcription']]

# Nettoyage du texte (minuscules, suppression des caractères spéciaux)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

# Appliquer le nettoyage et renommer les colonnes
df['Text Transcription'] = df['Text Transcription'].apply(clean_text)
df = df.rename(columns={'misogynous': 'label', 'Text Transcription': 'text'})

# Sauvegarder le nouveau CSV
df.to_csv("test.csv", index=False)
print("Fichier test_clean.csv créé avec succès !")
