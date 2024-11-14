# CHAPTER INFO: this chapter is all about the data and how to prepare the data for the model

# PART 0: load data and remove all rows with no data for the target-column

# PART 1: handle missing data

# PART 1.5: break the data into 4 parts for validation and training

# PART 2: handle categorical data (string data)

# PART 3: compare different models with the score_dataset function


#===================== Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.ensemble import RandomForestRegressor






# WAYS TO IMPROVE YOUR MODEL!
# STEP 1: comparing different modrandom forests models (model parameters)
# STEP 2: handle missing data (which data-removal-function)
# STEP 3: categorical variables




#======================== Categorical Varibles
# 1. das datenset erstellen/vorbereiten (missing values droppen, bei string kolonnen nur diese behalten die weniger als 20 verschiedene kathegrôrien haben)
# 2. das datenset verändern mit den 3 approaches (3 verschiedene datensets entstehen)
# den error für jeden approeach berechnen

# approach 1: string-kolonnen löschen (drop_X-data)
all_column_names = X.columns # alle kolonnennamen 
object_columns =  # liste mit erstellen wo colonnen-namen drin sind mit kategorischen inhalten
drop_data = # daten ohne die kategorischen kolonnen drin

# approach 2: Attention... all the different categories of the validation data needs to be in the trainind data!!!
# --> 2 listen mit kolonnen machen: kathegorien der kolonnen der validationdaten die in den trainingsdaten sind und solche die es nicht sind
# das encoding auf die liste mit den guten kolonnen andwenden
# delete die kolonnen mit den baden kathegorien (wäre es nicht besser nur die zeilen mit den schlechten werten zu löschen???)

# approach 3: Attention...cardinality (wie viele verschiedene kathegorien sind in einer bestimmten kolonne vertreten?) --> cardinality = anzahl neu erstellter kolonnen beim one-hot encoding!...man macht das one-hot encoding nur mit cardinality < 10
# --> 2 liste von kolonnen machen: high und low cardinality!
# --> das encoding auf die kolonnen der trainingsdaten anwenden die low cardinality haben
# --> delete kolonnen mit high cardinality

# approach 2 & 3 --> man kann gewisse zeilen mit ordinal encoding machen und gewisse zeilen mit dem one-hot-encoding









# define a function to measure each approach (dropping, ordinal, one-pot)
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)
    error = mae(prediction, y_valid)
    return error
