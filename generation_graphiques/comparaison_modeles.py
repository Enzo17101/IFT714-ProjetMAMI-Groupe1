import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# Import des fichiers CSV
result = "result_all_models.csv"

df = pd.read_csv(result, index_col=0)

# Extraction des valeurs pour la comparaison des modèles
test_acc_bayes = float(df.loc["test_acc", "bayes"])
test_f1_bayes = float(df.loc["test_f1_score", "bayes"])

test_acc_clip = float(df.loc["test_acc", "clip"])
test_f1_clip = float(df.loc["test_f1_score", "clip"])

test_acc_bert_clip = float(df.loc["test_acc", "bert+clip"])
test_f1_bert_clip = float(df.loc["test_f1_score", "bert+clip"])

test_acc_bert = float(df.loc["test_acc", "bert"])
test_f1_bert = float(df.loc["test_f1_score", "bert"])

test_acc_bert_blip = float(df.loc["test_acc", "bert+blip"])
test_f1_bert_blip = float(df.loc["test_f1_score", "bert+blip"])

# Catégories des mesures
models = ["Naive\nBayes", "Bert on\ncaption", "Bert on\ncaption + BLIP", "CLIP", "Bert on\ncaption + CLIP"]
metrics = ["Accuracy", "F1-score"]

# Valeurs
values = [
    [test_acc_bayes, test_f1_bayes],
    [test_acc_bert, test_f1_bert],
    [test_acc_bert_blip, test_f1_bert_blip],
    [test_acc_clip, test_f1_clip],
    [test_acc_bert_clip, test_f1_bert_clip]
]

# Position des barres
x = np.arange(len(models))
width = 0.45  # Largeur des barres

# Création de la figure et des axes
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, [v[0] for v in values], width, label="Accuracy", color="#ffa3c7")
rects2 = ax.bar(x + width/2, [v[1] for v in values], width, label="F1-score", color="#ff4c4c")

ax.set_ylim(0, 1)  # L"axe Y va de 0 à 1 (100%)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1))  # Affichage en pourcentage

# Labels et titre
ax.set_ylabel("Score (%)")
ax.set_title("Comparaison des modèles")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Affichage des valeurs au-dessus des barres
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height*100:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # Décalage vertical
                    textcoords="offset points",
                    ha="center", va="bottom")

plt.show()
fig.savefig("comparaison_all_models.png", dpi=300, bbox_inches="tight")
