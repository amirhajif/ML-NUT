import numpy as np
from collections import Counter

class KNN:
    def __init__(self,k=3):
        self.k=k

    def fit(self,X,y):
        self.X_train=X
        self.y_train=y

    def predict(self,X):
        predicted_labels=[self._predicts(x) for x in X]
        return np.array(predicted_labels)

    def _predicts(self,x):

        #calculate distance
        distance =[euclidean_distance(x,x_train) for x_train in self.X_train]

        # get K nearest
        k_indicies=np.argsort(distance)[:self.k]
        k_nearest_labels=[self.y_train[i] for i in k_indicies]

        # vote
        most_common=Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
