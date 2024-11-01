'''
Fragen:
- wäre es wohl besser gewesen anstatt utples ein dictionary zu verwenden?!
- eine website machen wo man ein model wählen kann, ein dataset und mit dingen wie anzahl nodes herumspielen
'''

# import 
import pandas as pd # wenn man etwas as importaet dann kann man einen namen definieren den man benutzt um das importierte aufzurufen falls der originalname zu lang oder kompliziert ist!
from sklearn.model_selection import train_test_split # funktion um die data zu splitten in trainings und validation data
from sklearn.tree import DecisionTreeRegressor # der modeltyp wird von da geholt
from sklearn.metrics import mean_absolute_error as mae # hiermit wird der mae berechnet


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
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=0, train_size=.75) # random_state makes sure that each round the same random set is chosen || train_size puts 75% into the training data and 25% in the testing data


#=======  Modell kreieren, fitten und predictions machen (allg.)  ========
# choose a model
model = DecisionTreeRegressor(random_state=0) # max_depth= ...als input kann man noch max_depth hineintun sobald man weiss wie viele knoten maximal der baum haben soll um under/overfitting zu vermeiden
# fit the model
model.fit(X_train, y_train)
# make predictions
prediction = model.predict(X_test)
# print(prediction)


#=======  Abweichung berechnen allg. (MAE)  ========
error = mae(prediction, y_test) # diese funktion berechnet den durchscnitt von allen differenzen zwischen prediction und dem eigentlichen targetwert der testdaten
# print(error)



# 1. Optimierung: Optimale Baumtiefe (anz. Nodes) finden



#======= Optimale Baumtiefe finden (funktion) ========
# Variablen
max_depth_values = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, None] # liste mit den kandidaten für die verschiedenen zu testenden baumtiefen; je kleiner der datensatz desto schneller kann overfitting passieren (overfitting = zu viele nodes)
depth_error_list = [] # leere liste aus tupln
depth_error_tuple = () # tuple bestehend aus kandidat für baumtiefe und seinem mae
beste_baumtiefe = None
kleinster_error = None
# resultate = ""
# (allg. funktion die den MAE berechnent in abhängigkeit der baumtiefe) --> man macht predictions mit allen kanidaten und beschliesst anhand des errors welcher der der kleinste ist
def MAE(liste_mit_baumtiefen):
    for baumtiefe in liste_mit_baumtiefen:
            model = DecisionTreeRegressor(max_depth_value=baumtiefe)
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            error = mae(prediction, y_test)
            if error < kleinster_error:
                 kleinster_error = error
                 beste_baumtiefe = baumtiefe 
            # depth_error_tuple = (baumtiefe, error)
            # depth_error_list = depth_error_list.append(depth_error_tuple)
            # resultate = f"Baumtiefe:{baumtiefe}, Error:{error}" # resultate für jeden loop herausprinten!
            # print(resultate)
    # print(depth_error_list)
    return kleinster_error, beste_baumtiefe # nun kann man beste_baumtiefe in das model als parameter für max_depth_value=beste_baumtiefe








    


