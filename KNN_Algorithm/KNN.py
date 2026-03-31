from collections import Counter

import numpy as np

def euclidean_distance(x1, x2):
    dis = np.sqrt(np.sum((x1-x2)**2))
    return dis 



class KNN:
    def __init__(self, k=3):
        self.k = k # liczba sasiadow

    
    def fit(self, X, y):
        self.X_train = X 
        self.y_train = y 
    
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    

    def _predict(self, x):
        
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(distances)[: self.k]
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]
    



if __name__ == "__main__":
   
    import pandas as pd
    import numpy as np
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

   
    cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

   
    iris = datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    
 
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

