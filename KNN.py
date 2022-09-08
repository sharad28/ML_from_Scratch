import numpy as np
class KNN:
    def __init__(self,k):
        #defining # nearest neighbors
        self.k=k
    
    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def _distance(self, x):
        return np.sqrt(sum(X_train-x)**2)

    def _predict(self,x_test):
        pass


    def predict(self,X_test):
        predict_list = []
        pass
if __name__=="__main__":
    print('hello')        