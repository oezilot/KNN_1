'''
Fragen:
- wäre es wohl besser gewesen anstatt utples ein dictionary zu verwenden?!
- eine website machen wo man ein model wählen kann, ein dataset und mit dingen wie anzahl nodes herumspielen
- ein gntm machen statt mit menschenmodels languagemodels etc (bodytype=modeltype)
- ausprobieren: ich möchte eine prediction für ein bestimmtes rezept machen (im terminal kann man dann quasi ein formular ausfüllen mit zutaten mit welcher dann eine prediction gemacht wird zum rating)


Meine verschiedenen Models:
- decisiontreeregressor (mit und ohne max_depth_value)
- randomforestregressor (mit oder ohne max_depth_value?!)
'''


#===================== Import necessary libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.ensemble import RandomForestRegressor


#===================== Load and Read Data

dataset_path = "data/melb_data.csv"
dataset = pd.read_csv(dataset_path)

# Print Data
# print(dataset.head())  # Uncomment to explore data

# Target and Features (VERBESSERUNGSPOTENTIAL)
target = dataset['Price']
feature_titles = ['Rooms', 'Type', 'Bathroom', 'Bedroom2', 'Landsize', 'BuildingArea', 'Car', 'YearBuilt', 'CouncilArea', 'Regionname']
features = pd.get_dummies(dataset[feature_titles], drop_first=True)  # One-hot encoding

# Split Data into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=0, train_size=.75)


#===================== Error-function (VERBESSERUNGPOTENTIAL)

# Initialize global variables for MAE optimization
kleinster_error = float('inf')  # Start with infinity to find minimum
beste_baumtiefe = None

# Function to find optimal depth (VERBESSERUNSPOTNTIAL BEI DER WAHL DER VERSCHIENENEN BAUMTIEFEN)...diese function berechnet die prediction zu verschiedenen baumtiefen und vergleicht diese um die beste baumtiefe herauszufinden
def MAE(liste_mit_baumtiefen, X_train, y_train, X_test, y_test):
    global kleinster_error, beste_baumtiefe, prediction1b
    # error für jede baumtiefe berechnen
    for baumtiefe in liste_mit_baumtiefen:
        model = DecisionTreeRegressor(max_depth=baumtiefe, random_state=0)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        error = mae(prediction, y_test)
        # die resultate der verschiedenen baumtiefen vergleichen
        if error < kleinster_error:
            kleinster_error = error
            beste_baumtiefe = baumtiefe
            prediction1b = prediction

# Run MAE optimization
max_depth_values = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, None]
MAE(max_depth_values, X_train, y_train, X_test, y_test)


#===================== vershiedene Models trainieren und fitten, predictions

# Model ohne Optimierung
model = DecisionTreeRegressor(random_state=1)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
error = mae(prediction, y_test)

# Train Decision Tree with optimal depth 
model1 = DecisionTreeRegressor(max_depth=beste_baumtiefe, random_state=0)
model1.fit(X_train, y_train)
prediction1 = model1.predict(X_test)
error1 = mae(prediction1, y_test)

# Train Random Forest
model2 = RandomForestRegressor(random_state=12)
model2.fit(X_train, y_train)
prediction2 = model2.predict(X_test)
error2 = mae(prediction2, y_test)


#===================== Resultate definieren und printen

# allg. wie gut ist das model testen
# Die errors der modelle in einem Dictionary übersichtlich speichern und vergleichen 
error_dictionary = {}

error_dictionary['Decision-Tree'] = float(error)
error_dictionary['Decision-Tree-Optimized-Nodes'] = float(error1)
error_dictionary['Random-Forest'] = float(error2)


# eine prediction für spezifische werte der parameter


# error-dictionary printen
print(f"Model Errors: {error_dictionary}")




