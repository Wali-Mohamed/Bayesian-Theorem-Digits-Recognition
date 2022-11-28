import numpy as num
def accuracy(y, y_hat):
    return num.mean(y==y_hat)

class MinMaxScaler():
    
    def fit(self, X):
        result=((X-X.min())/(X.max()-X.min()))
        
        return result
        
    def transform(self, X):
        
        
        return self.result





def MinMaxScaler2(X):
    result=((X-X.min())/(X.max()-X.min()))
    return result