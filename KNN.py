import numpy as np
from collections import Counter
class KNN:
    def __init__(self,k):
        #defining # nearest neighbors
        self.k=k
    
    def fit(self,X_train,y_train):
        self.X_train = np.array(X_train)
        self.y_train = y_train
    
    def _euclidean_distance(self, x_test,x_train):
        return np.sqrt(np.sum((x_train-x_test)**2))

    def _predict(self,x_test):
        dist=[self._euclidean_distance(x_test,x_train) for x_train in self.X_train]
        dist_idx = np.argsort(dist)[:self.k]
        labels = [self.y_train[idx] for idx in dist_idx]
        common_count = Counter(labels)
        common_count=common_count.most_common(1)
        # print(common_count[0][0])
        return common_count[0][0]

    def predict(self,X_test):
        predict_list = [self._predict(x) for x in X_test]
        return predict_list

if __name__ == "__main__":
    def accuracy(y_true, y_pred):
        if len(y_pred)==len(y_test):
            count = 0
            for i in range(len(y_pred)):
                if y_pred[i]==y_test[i]:
                    count += 1
        accuracy = count / len(y_true)
        return accuracy

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    k = 5
    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(len(predictions))
    print(len(y_test))
    print("KNN classification accuracy", accuracy(y_test, predictions))
    