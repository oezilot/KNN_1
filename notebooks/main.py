'''
Fragen:
- wäre es wohl besser gewesen anstatt utples ein dictionary zu verwenden?!
- eine website machen wo man ein model wählen kann, ein dataset und mit dingen wie anzahl nodes herumspielen
- ein gntm machen statt mit menschenmodels languagemodels etc (bodytype=modeltype)

Meine verschiedenen Models:
- decisiontreeregressor (mit und ohne max_depth_value)
- randomforestregressor (mit oder ohne max_depth_value?!)
'''

# import 
import pandas as pd # wenn man etwas as importaet dann kann man einen namen definieren den man benutzt um das importierte aufzurufen falls der originalname zu lang oder kompliziert ist!
from sklearn.model_selection import train_test_split # funktion um die data zu splitten in trainings und validation data
from sklearn.tree import DecisionTreeRegressor # der modeltyp wird von da geholt
from sklearn.metrics import mean_absolute_error as mae # hiermit wird der mae berechnet
from sklearn.ensemble import RandomForestRegressor 


# resultatssheet definieren: für jede art des models werden die resultate übersichtlich herausgprintet und in einer gewissen darstellung sichtbar und vergleichbar gemacht (ein dictionary hier machen?)
resultate1a = None # tree_prediction
resultate1b = None
resultate2 = None # forest_prediction


#======= Load and read Data  ========
# load data
dataset_path = "data/melb_data.csv"
# read data
dataset = pd.read_csv(dataset_path)


#======= Print Data (explore data)  ========
# print first couple of rows
# print(dataset.head())
# die ersten 3 zeilen und  kolonnen herausprinten
#print(dataset.iloc[:, :3].head())

# informationen (dateigrösse/memory usage, anz. kolonnen und zeilennamen, dtypes)
# print(dataset.info())


#=======  Daten organisieren/aufteilen  ========
# Target
target = dataset['Price']
# Features
features = ['Rooms', 'Type', 'Bathroom', 'Bedroom2','Landsize', 'BuildingArea', 'Car', 'YearBuilt' ,'CouncilArea', 'RegionName']
# 4 Datentypen-Aufteilung (diese funktion nimmt features und target und unterteilt diese in jeweils 75 und 25 häufchen, zurück wird ein multiple gegeben)
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=0, train_size=.75) # random_state makes sure that each round the same random set is chosen (each number for random_state=number stands for a different randomness) || train_size puts 75% into the training data and 25% in the testing data


#=======  Modell kreieren, fitten und predictions machen (allg.)  ========
# choose a model
model1 = DecisionTreeRegressor(random_state=0) # random_state=number: randomness wird verwendet bei der konstellation der verschiedenen datenpunkten || max_depth= ...als input kann man noch max_depth hineintun sobald man weiss wie viele knoten maximal der baum haben soll um under/overfitting zu vermeiden
# fit the model
model1.fit(X_train, y_train)
# make predictions
prediction1a = model1.predict(X_test)
# print(prediction)


#=======  Abweichung berechnen allg. (MAE)  ========
error1a = mae(prediction1a, y_test) # diese funktion berechnet den durchscnitt von allen differenzen zwischen prediction und dem eigentlichen targetwert der testdaten
# print(error)



# 1. Optimierung: Optimale Baumtiefe (anz. Nodes) finden



#======= Optimale Baumtiefe finden (funktion) ========
# Variablen
max_depth_values = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, None] # liste mit den kandidaten für die verschiedenen zu testenden baumtiefen; je kleiner der datensatz desto schneller kann overfitting passieren (overfitting = zu viele nodes)
depth_error_list = [] # leere liste aus tupln
depth_error_tuple = () # tuple bestehend aus kandidat für baumtiefe und seinem mae
beste_baumtiefe = None
kleinster_error = 100000000000000000000 # das könnte man auch schöner machen hier!!!
prediction1b = None
# resultate = ""
# (allg. funktion die den MAE des model1 berechnent in abhängigkeit der baumtiefe) --> man macht predictions mit allen kanidaten und beschliesst anhand des errors welcher der der kleinste ist
def MAE(liste_mit_baumtiefen, X_train, y_train, X_test, y_test):
    for baumtiefe in liste_mit_baumtiefen:
            model = DecisionTreeRegressor(max_depth_value=baumtiefe, random_state=0)
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            error = mae(prediction, y_test)
            if error < kleinster_error:
                 kleinster_error = error
                 beste_baumtiefe = baumtiefe
                 prediction1b = prediction 
            # depth_error_tuple = (baumtiefe, error)
            # depth_error_list = depth_error_list.append(depth_error_tuple)
            # resultate = f"Baumtiefe:{baumtiefe}, Error:{error}" # resultate für jeden loop herausprinten!
            # print(resultate)
    # print(depth_error_list)
    return kleinster_error, beste_baumtiefe, prediction1b # nun kann man beste_baumtiefe in das model als parameter für max_depth_value=beste_baumtiefe


# 2. Optimierung: Random forests (besserer modeltyp; um die gesamtprediction des tarets zu berechnen wird das mittel von mehreren verschiedenen trees verwendet)
# model definieren, fitten, predicten
model2 = RandomForestRegressor(random_state=12) # braucht es hier auch max_depth_values?
model2.fit(X_train, y_train)
prediction2 = model2.predict(X_test)
error2 = mae(prediction2, y_test)



# Resultate (Verschiedene Models vergleichen!)
resultat1a = f"Tree ohne Optimierung:{prediction1a}, Error:{error1a}"
resultat1b = f"Tree mit Optimierung:{prediction1b}, Error:{kleinster_error}"
resultat2 = f"Forest:{prediction2}. Error:{error2}"
print(resultat1a, resultat1b, resultat2)

