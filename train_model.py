# Import des bibliothèques nécessaires
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib  # Pour sauvegarder le modèle

# Chargement du fichier CSV contenant les données
df = pd.read_csv("base_finale.csv", sep=";")  # Le séparateur est un point-virgule

# Séparation des données en variables explicatives (X) et variable cible (y)
X = df.drop("prix", axis=1)  # Toutes les colonnes sauf 'prix'
y = df["prix"]               # Colonne à prédire : le prix immobilier

# Division des données en jeu d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Création d'un pipeline de traitement automatique
model = make_pipeline(
    SimpleImputer(strategy="mean"),              # Remplace les valeurs manquantes par la moyenne
    GradientBoostingRegressor(n_estimators=100,  # 100 arbres
                              learning_rate=0.1, # Taux d'apprentissage
                              max_depth=3,       # Profondeur des arbres
                              random_state=42)   # Reproductibilité
)

# Entraînement du modèle sur les données d'entraînement
model.fit(X_train, y_train)

# Sauvegarde du modèle entraîné dans un fichier .pkl
joblib.dump(model, "model.pkl")

# Affichage des noms de colonnes utilisées (utile pour la prédiction ensuite)
print(X_train.columns)

# Message de confirmation
print("Modèle Gradient Boosting entraîné avec succès ✅")
