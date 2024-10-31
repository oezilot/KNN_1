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

#======= Load Data  ========
# load data
dataset_path = "data/melb_data.csv"
# read data
dataset = pd.read_csv(dataset_path)


#======= Print Data  ========
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
# 4 Datentypen-Aufteilung (diese funktio nimmt input und output und unterteilt diese in jeweils 75 und 25 häufchen)
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=0, train_size=.75) # random_state makes sure that each round the same random set is chosen || train_size puts 75% into the training data and 25% in the testing data


#=======  Modell kreieren, fitten und predictions machen (allg.)  ========
# choose a model
model = DecisionTreeRegressor(random_state=0) # max_depth= ...als input kann man noch max_depth hineintun sobald man weiss wie viele knoten maximal der baum haben soll um under/overfitting zu vermeiden
# fit the model
model.fit(X_train, y_train)
# make predictions
prediction = model.predict(X_test)
# print(prediction)


#=======  Abweichung berechnen (MAE)  ========
error = mae(prediction, y_test) # diese funktion berechnet den durchscnitt von allen differenzen zwischen prediction und dem eigentlichen targetwert der testdaten
# print(error)


#=======  Optimale Anzahl Nodes finden (Tiefe des Baumes optimieren / anzahl blätter) ========
# verschiedene anz blätter für verschiedene tiefen des baumes
max_depth_values = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, None] # je kleiner der datensatz desto schneller kann overfitting passieren (overfitting = zu viele nodes)
depth_error_list = [] # leere liste mit tuple aus der baumtiefe und ihrem error
depth_error_tuple = ()
resultate = ""
optimal_depth = None
# den error zu verschiedenen baumtiefen berechnen (allg. funktion die den MAE berechnent in abhängigkeit der baumtiefe)
def MAE(liste_mit_baumtiefen):
    for baumtiefe in liste_mit_baumtiefen:
            model = DecisionTreeRegressor(max_depth_value=baumtiefe)
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            error = mae(prediction, y_test)
            depth_error_tuple = (baumtiefe, error)
            depth_error_list = depth_error_list.append(depth_error_tuple)
            # resultate = f"Baumtiefe:{baumtiefe}, Error:{error}" # resultate für jeden loop herausprinten!
            # print(resultate)
    print(depth_error_list)
    return depth_error_list
# die verschiedenen baumtiefen vergleichen und anhand ihres errors auswerten (je kleiner der error desto besser ist die baumtiefe!)

# diese funktion returnt den wert (erstes tupleelement) des tiefsten errors (2tes tupleelement) zurück
def optimal(liste_mit_tuples):
    # Erzeuge eine Liste aller Fehlerwerte (zweite Elemente der Tupel)
    alle_errors = [i[1] for i in liste_mit_tuples]
    
    # Finde den kleinsten Fehlerwert
    min_error = min(alle_errors)
    
    # Finde das erste Tupel, bei dem der zweite Wert (Fehler) gleich dem minimalen Fehler ist
    for tupel in liste_mit_tuples:
        if tupel[1] == min_error:
            return tupel[0]  # Gib das erste Element dieses Tupels zurück

optimal_depth = optimal(depth_error_list)
print(optimal_depth)






    


