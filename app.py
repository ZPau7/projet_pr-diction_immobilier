import pandas as pd
from flask import Flask, render_template, request
import joblib

# Charger le modèle pré-entraîné
model = joblib.load('model.pkl')

# Créer une fonction pour calculer le poids de l'arrondissement
def poids_arrondissement(val):
    if pd.isna(val):
        return None
    val = str(val).strip().lower().replace("é", "e")

    arrdt = {
        "arrdt_75016": 0.196009, "arrdt_75008": 0.181532, "arrdt_75007": 0.117580,
        "arrdt_75006": 0.081818, "arrdt_75017": 0.053731, "arrdt_75015": 0.052894,
        "arrdt_75004": 0.052450, "arrdt_75002": 0.048162, "arrdt_75003": 0.030506,
        "arrdt_75001": 0.030025, "arrdt_94160": 0.027274, "arrdt_75009": 0.023671,
        "arrdt_75011": 0.020369, "arrdt_75014": 0.019963, "arrdt_94300": 0.019930,
        "arrdt_75005": 0.016535, "arrdt_75012": 0.014442, "arrdt_94220": 0.014016,
        "arrdt_75010": 0.012957, "arrdt_94210": 0.004845, "arrdt_94205": -0.001349,
        "arrdt_75013": -0.001420, "arrdt_75020": -0.001544, "arrdt_94340": -0.003944,
        "arrdt_75019": -0.009978, "arrdt_94130": -0.011403, "arrdt_75018": -0.011981,
        "arrdt_94520": -0.014637, "arrdt_94250": -0.015446, "arrdt_94490": -0.015731,
        "arrdt_94410": -0.015753, "arrdt_94100": -0.016114, "arrdt_94170": -0.016886,
        "arrdt_94880": -0.017151, "arrdt_94150": -0.021716, "arrdt_94380": -0.022601,
        "arrdt_94700": -0.022604, "arrdt_94460": -0.024080, "arrdt_94110": -0.024309,
        "arrdt_94360": -0.024764, "arrdt_94240": -0.025032, "arrdt_94270": -0.025259, "arrdt_94480": -0.025304,
        "arrdt_94510": -0.025782, "arrdt_94440": -0.026175, "arrdt_94550": -0.026732, "arrdt_94230": -0.027300,
        "arrdt_94120": -0.030759, "arrdt_94290": -0.035054, "arrdt_94370": -0.037431, "arrdt_94420": -0.040048,
        "arrdt_94310": -0.045189, "arrdt_94450": -0.046450, "arrdt_94800": -0.047676, "arrdt_94140": -0.048839,
        "arrdt_94320": -0.048982, "arrdt_94470": -0.049155, "arrdt_94260": -0.049526, "arrdt_94350": -0.050313,
        "arrdt_94200": -0.055588, "arrdt_94190": -0.057833, "arrdt_94430": -0.057909, "arrdt_94400": -0.059997,
        "arrdt_94600": -0.060298, "arrdt_94500": -0.075109
    }

    for key in arrdt:
        if key in val:
            return arrdt[key]
    
    return None

# Création de l'application Flask
app = Flask(__name__)

# Route d'accueil
@app.route('/')
def home():
    return render_template('index.html')

# Route pour la prédiction du prix
@app.route('/predict', methods=['POST'])
def predict():
    superficie = float(request.form['superficie'])
    pieces = int(request.form['pieces'])
    chambres = int(request.form['chambres'])
    arrondissement = request.form['arrondissement']
    cave = 1 if 'cave' in request.form else 0
    ascenseur = 1 if 'ascenseur' in request.form else 0
    gardien = 1 if 'gardien' in request.form else 0

    arrdt_value = poids_arrondissement(arrondissement)

    
    input_data = pd.DataFrame([[pieces, chambres, ascenseur, cave, gardien, superficie, arrdt_value]],
                              columns=['Nombre de pieces', 'Nombre de chambres', 'Ascenseur', 'Cave', 'Gardien', 'superficie', 'arrondissement'])

    predicted_price = model.predict(input_data)[0]
    prediction_percentage = 80

    prediction_text = f"Le prix estimé est de {predicted_price:,.2f} €"

    return render_template('index.html', prediction_text=prediction_text, prediction_percentage=prediction_percentage)

@app.route("/about")
def about():
    return render_template("about-us.html")
# Lancer l'application Flask
if __name__ == "__main__":
    import os
    app.run(debug=True, port=5001, use_reloader=os.environ.get("WERKZEUG_RUN_MAIN") == "true")
