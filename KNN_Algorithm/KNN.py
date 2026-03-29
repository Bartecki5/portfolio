from collections import Counter

import numpy as np

def euclidean_distance(x1, x2):
    dis = np.sqrt(np.sum((x1-x2)**2))
    return dis 

#x1 = np.array([1,2,0])
#x2 = np.array([4,6,12])
#print(euclidean_distance(x1,x2))

class KNN:
    def __init__(self, k=3):
        self.k = k # liczba sasiadow

    #ladowanie danych do magazynu
    def fit(self, X, y):
        self.X_train = X 
        self.y_train = y 
    
    #bierze ciezarowke nowych paczek
    #przekazuje kazda z nich po kolei do
    #pracownika (_predict)
    #zbiera od niego etykiety
    #oddaje ci gotowy raport
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    

    def _predict(self, x):
        #compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        #sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[: self.k]
        #Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        #return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]
    



if __name__ == "__main__":
    # Niezbędne importy
    import pandas as pd
    import numpy as np
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    # Opcjonalna mapa kolorów, jeśli planujesz później robić wykresy
    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

   
    iris = datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    
    # Opcjonalnie: wyświetlenie kształtu tabeli i pierwszych wierszy
    print("Kształt danych w Pandas:", df.shape)
    print("Pierwsze 5 wierszy tabeli:\n", df.head(), "\n")

    
    X = df.drop('species', axis=1).values 
    y = df['species'].values              

   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )


    k = 3
    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    

    print("KNN classification accuracy:", accuracy(y_test, predictions))

