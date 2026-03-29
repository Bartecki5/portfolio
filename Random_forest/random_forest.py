from collections import Counter
import numpy as np
from decision_tree import DecisionTree

def bootstrap_sample(X,y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace = True)
    return X[idxs], y[idxs]

def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common

class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_feats = None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X,y):
        self.trees = []

        if self.n_feats is None:
            self.n_feats = int(np.sqrt(X.shape[1]))


        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_feats = self.n_feats,
            )
            X_samp, y_samp = bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)
             
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0 ,1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)

        
if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import random 
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    
    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


    tree = DecisionTree(max_depth=10)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)
    acc_tree = accuracy(y_test, y_pred_tree)
    print(f"Pojedyncze drzewo {acc_tree}")


    # SKLEARN DRZEWO
    sk_tree = DecisionTreeClassifier(max_depth=10, random_state=1234)
    sk_tree.fit(X_train, y_train)
    sk_acc_tree = accuracy(y_test, sk_tree.predict(X_test))
    print(f"Pojedyncze drzewo (SKLEARN) : {sk_acc_tree:.4f}")


    best_accuracy = 0
    best_params = {}
    n_iterations = 15 




    for i in range(n_iterations):
       
        test_n_trees = random.randint(5, 50)           
        test_max_depth = random.randint(3, 15)         
        test_min_samples = random.randint(2, 10)    

        
        clf_random = RandomForest(
            n_trees=test_n_trees, 
            max_depth=test_max_depth, 
            min_samples_split=test_min_samples
        )
        
  
        clf_random.fit(X_train, y_train)
        y_pred_random = clf_random.predict(X_test)
        acc_random = accuracy(y_test, y_pred_random)

        print(f"Próba {i+1}/{n_iterations} | n_trees: {test_n_trees:2}, max_depth: {test_max_depth:2}, min_samples: {test_min_samples:2} --> Wynik: {acc_random:.4f}")


        if acc_random > best_accuracy:
            best_accuracy = acc_random
            best_params = {
                "n_trees": test_n_trees,
                "max_depth": test_max_depth,
                "min_samples_split": test_min_samples
            }

    print(f"Najlepsze znalezione parametry: {best_params}")
    print(f"Najwyższe Accuracy: {best_accuracy:.4f} ({(best_accuracy*100):.2f}%)")

    sk_forest = RandomForestClassifier(n_estimators=100, random_state=1234)
    sk_forest.fit(X_train, y_train)
    sk_acc_forest = accuracy(y_test, sk_forest.predict(X_test))
    print(f"Las Losowy (SKLEARN)        : {sk_acc_forest:.4f} ({(sk_acc_forest*100):.2f}%)\n")








