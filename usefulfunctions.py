import numpy as num
def accuracy(y, y_hat):
    return num.mean(y==y_hat)

class MinMaxScaler():
    
    
        
        
    def transform(self, X):
        
        result=((X-X.min())/(X.max()-X.min()))
        
        return result
        





def MinMaxScaler2(X):
    result=((X-X.min())/(X.max()-X.min()))
    return result