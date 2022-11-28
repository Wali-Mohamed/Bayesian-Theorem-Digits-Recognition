import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvn

class GaussNB():
    
    def fit(self, X, y, epsilon=1e-3):
        self.likelihoods=dict()
        self.priors = dict()
        self.K = set(y.astype(int))
    
        for k in self.K:
            
            X_k = X[y==k,:]
            
            self.likelihoods[k]={'mean':X_k.mean(axis=(0)), 'cov': X_k.var(axis=(0)) + epsilon}
            self.priors[k] = len(X_k)/len(X)
        
    def predict(self, X):
        N,D = X.shape
        P_hat =np.zeros((N,len(self.K)))
        
        
        
        for k, l in self.likelihoods.items():
            
            # Apply Bayes Theorem
            P_hat[:,k]= mvn.logpdf(X, l['mean'],l['cov'])+np.log(self.priors[k])
        
        return P_hat.argmax(axis=1) 

class Gauss():
    def fit(self, X, y, epsilon=1e-3):
        self.likelihoods=dict()
        self.priors = dict()
        self.K = set(y.astype(int))
    
        for k in self.K:
            
            X_k = X[y==k,:]

            N_k, D = X_k.shape

            Mu_k = X_k.mean(axis=0)
            
            self.likelihoods[k]={'mean':Mu_k, 'cov':(1/(N_k-1))*np.matmul((X_k-Mu_k).T,X_k-Mu_k)  + epsilon*np.identity(D)}
            self.priors[k] = len(X_k)/len(X)
            
    def predict(self, X):
        N, D = X.shape
        P_hat =np.zeros((N, len(self.K)))
        
        
        
        for k, l in self.likelihoods.items():
            
            # Apply Bayes Theorem
            P_hat[:,k]=mvn.logpdf(X, l['mean'],l['cov'])+np.log(self.priors[k])
        
        return P_hat.argmax(axis=1) 

    
class KNNclassifier():
    def fit(self, X, y):
        self.X=X
        self.y=y
    def predict(self, X, k , epsilon=1e-3):
        N = len(X)
        y_hat = np.zeros(N)
        for i in range(N):
              dist2 = np.sum((self.X-X[i])**2, axis=1)
              idxt = np.argsort(dist2)[:k]
              gamma_k =1/np.sqrt(dist2[idxt]+epsilon)
              y_hat[i] =np.bincount(self.y[idxt], weights=gamma_k).argmax()
        return y_hat
