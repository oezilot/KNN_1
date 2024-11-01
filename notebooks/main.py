'''
Fragen:
- wäre es wohl besser gewesen anstatt utples ein dictionary zu verwenden?!
- eine website machen wo man ein model wählen kann, ein dataset und mit dingen wie anzahl nodes herumspielen
- ein gntm machen statt mit menschenmodels languagemodels etc (bodytype=modeltype)

Meine verschiedenen Models:
- decisiontreeregressor (mit und ohne max_depth_value)
- randomforestregressor (mit oder ohne max_depth_value?!)
'''

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.ensemble import RandomForestRegressor

# Variables to hold results
resultat1 = None
resultate2 = None

# Load and read Data
dataset_path = "data/melb_data.csv"
dataset = pd.read_csv(dataset_path)

# Print Data
# print(dataset.head())  # Uncomment to explore data

# Target and Features
target = dataset['Price']
feature_titles = ['Rooms', 'Type', 'Bathroom', 'Bedroom2', 'Landsize', 'BuildingArea', 'Car', 'YearBuilt', 'CouncilArea', 'Regionname']
features = pd.get_dummies(dataset[feature_titles], drop_first=True)  # One-hot encoding

# Split Data
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=0, train_size=.75)

# Initialize global variables for MAE optimization
kleinster_error = float('inf')  # Start with infinity to find minimum
beste_baumtiefe = None
prediction1b = None

# Function to find optimal depth
def MAE(liste_mit_baumtiefen, X_train, y_train, X_test, y_test):
    global kleinster_error, beste_baumtiefe, prediction1b
    for baumtiefe in liste_mit_baumtiefen:
        model = DecisionTreeRegressor(max_depth=baumtiefe, random_state=0)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        error = mae(prediction, y_test)
        if error < kleinster_error:
            kleinster_error = error
            beste_baumtiefe = baumtiefe
            prediction1b = prediction

# Run MAE optimization
max_depth_values = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, None]
MAE(max_depth_values, X_train, y_train, X_test, y_test)

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

# Print results
average_prediction1 = prediction1.mean()  # Durchschnitt aller Vorhersagen
average_prediction2 = prediction2.mean()  # Durchschnitt aller Vorhersagen
resultat1 = f"Tree mit Optimierung: {average_prediction1}, Error: {error1}"
resultat2 = f"Forest: {average_prediction2}, Error: {error2}"
print(resultat1) # das gibt eine liste zurück mir den predictions für jede zeile
print("")
print(resultat2)

# ausprobieren: ich möchte eine prediction für ein bestimmtes rezept machen (im terminal kann man dann quasi ein formular ausfüllen mit zutaten mit welcher dann eine prediction gemacht wird zum rating)
