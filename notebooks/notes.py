#======= Tuple-Analyse ========

list_of_tuples = [(1, 2), (3, 4), (5, 6)]
#print(list_of_tuples[1])

#print(list_of_tuples)

# das erste element des ersten tuples
#print(list_of_tuples[0][0])

# das erste element von jedem tuple
for i in list_of_tuples:
    print(i[0])


max_depth_values = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, None] # liste mit den kandidaten für die verschiedenen zu testenden baumtiefen; je kleiner der datensatz desto schneller kann overfitting passieren (overfitting = zu viele nodes)
depth_error_list = []
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
