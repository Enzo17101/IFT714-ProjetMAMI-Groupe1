import pandas as pd
import numpy as np
import csv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

# Chargement des données
train_df = pd.read_csv("../preprocessing_with_BLIP/train_blip.csv")
test_df = pd.read_csv("../preprocessing_with_BLIP/test_blip.csv")

# Nettoyage des données
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# On fusionne les données (texte + description) et labels
train_texts = train_df["text"]  # + " " + train_df["img_desc"]
test_texts = test_df["text"]  # + " " + test_df["img_desc"]
train_labels = train_df["label"]
test_labels = test_df["label"]

# Vectorisation TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # On limite le vocabulaire
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Entraînement du modèle Naïve Bayes
model = MultinomialNB()
model.fit(X_train, train_labels)

# Prédictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Évaluation
train_acc = accuracy_score(train_labels, train_preds)
test_acc = accuracy_score(test_labels, test_preds)
train_f1 = f1_score(train_labels, train_preds, average='binary')
test_f1 = f1_score(test_labels, test_preds, average='binary')

# Distribution des prédictions
def compute_distribution(y_pred):
    unique, counts = np.unique(y_pred, return_counts=True)
    return {label: int(count) for label, count in zip(unique, counts)}


train_counts = compute_distribution(train_preds)
test_counts = compute_distribution(test_preds)


# Fonction pour formater les dictionnaires pour CSV
def format_dict(d):
    return " , ".join(f"{k} : {v}" for k, v in d.items())


# Hyperparamètres et résultats
params = {
    "vectorizer": "TF-IDF",
    "max_features": 5000,
    "classifier": "MultinomialNB",
    "train_acc": round(train_acc, 4),
    "train_f1_score": round(train_f1, 4),
    "test_acc": round(test_acc, 4),
    "test_f1_score": round(test_f1, 4),
    "train_pred_distribution": format_dict(train_counts),
    "test_pred_distribution": format_dict(test_counts)
}

output_file = "naive_bayes_results.csv"

# Crée le fichier s'il n'existe pas, le met à jour sinon
if os.path.exists(output_file):
    with open(output_file, mode='r', newline='', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        rows = list(reader)

    header = rows[0]
    existing_value_columns = [col for col in header[1:] if col.startswith("valeurs_")]
    new_col_name = f"valeurs_{(max([int(col.split('_')[1]) for col in existing_value_columns]) + 1) if existing_value_columns else 1}"

    # Nouvelle colonne
    header.append(new_col_name)
    for i in range(1, len(rows)):
        key = rows[i][0]
        rows[i].append(params.get(key, ""))

    # Nouvelles données
    with open(output_file, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
else:
    # Nouveau fichier
    with open(output_file, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(["Hyperparamètres et Résultats", "valeurs_1"])
        for key, value in params.items():
            writer.writerow([key, value])

print(f"Données exportées avec succès dans {output_file}!")
