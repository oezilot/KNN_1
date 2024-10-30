# import 
import pandas as pd # wenn man etwas as importaet dann kann man einen namen definieren den man benutzt um das importierte aufzurufen falls der originalname zu lang oder kompliziert ist!


#======= Load Data  ========
# load data
dataset_path = "data/melb_data.csv"
dataset = pd.read_csv(dataset_path)

#======= Print Data  ========
# print first couple of rows
print(dataset.head())
# die ersten 3 zeilen und kolonnen herausprinten
#print(dataset.iloc[:, :3].head())

# informationen (dateigrösse/memory usage, anz. kolonnen und zeilen, dtypes)
print(dataset.info())

# welches merkmal möchte man predicten? (=target)
#print(dataset['Price']) # den resi für alle rows printen

# welche merkmale könnten desen presi beeinflussen? (=features)
features = ['Rooms', 'Type', 'Date', 'Bathroom', 'Bedroom2', 'Car', 'YearBuilt' ,'CouncilArea']
#print(dataset[features])