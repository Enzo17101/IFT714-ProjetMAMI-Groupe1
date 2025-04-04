import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# Import des fichiers CSV
result = "../Model/hyperparameters_results.csv"

df = pd.read_csv(result, index_col=0)

# Extraction des valeurs pour la comparaison des modèles
test_acc_rl = float(df.loc["test_acc", "rl"])
test_f1_rl = float(df.loc["test_f1_score", "rl"])

test_acc_rf = float(df.loc["test_acc", "rf"])
test_f1_rf = float(df.loc["test_f1_score", "rf"])

test_acc_rn = float(df.loc["test_acc", "rn"])
test_f1_rn = float(df.loc["test_f1_score", "rn"])

# Catégories des mesures
models = ["Régression\nlogistique", "Random\nforest", "Neural\nnetwork"]
metrics = ["Accuracy", "F1-score"]

# Valeurs
values = [
    [test_acc_rl, test_f1_rl],
    [test_acc_rf, test_f1_rf],
    [test_acc_rn, test_f1_rn]
]

# Position des barres
x = np.arange(len(models))
width = 0.4  # Largeur des barres

# Création de la figure et des axes
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, [v[0] for v in values], width, label="Accuracy", color="#ffa3c7")
rects2 = ax.bar(x + width/2, [v[1] for v in values], width, label="F1-score", color="#ff4c4c")

ax.set_ylim(0, 1)  # L"axe Y va de 0 à 1 (100%)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1))  # Affichage en pourcentage

# Labels et titre
ax.set_ylabel("Score (%)")
ax.set_title("Comparaison des méthodes avec CLIP")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Affichage des valeurs au-dessus des barres
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.2%}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # Décalage vertical
                    textcoords="offset points",
                    ha="center", va="bottom")

plt.show()
fig.savefig("comparaison_CLIP.png", dpi=300, bbox_inches="tight")
