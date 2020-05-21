import numpy as np 
from scipy.stats import multivariate_normal

def GDA(X , y):
    m , n = X.shape # m - training  examples , n - features
    classes = np.unique(y)
    num_classes = len(classes)

    phi = np.zeros((num_classes , 1)) # probablity of getting each class
    means = np.zeros((num_classes , n)) # means of x vector for each class
    sigma = 0

    for i in range(num_classes):
        ind = np.flatnonzero(y == classes[i])
        phi[i] = len(ind)/m
        means[i] = np.mean(X[ind] , axis = 0)
        sigma += np.cov(X[ind].T)*(len(ind) - 1)
    
    sigma = sigma/m
    return means , sigma , phi , classes

def pdf(X , mean , sigma):
    var = multivariate_normal.pdf(X , mean = mean , cov = sigma)
    return var

def predict(X , means , sigma , phi , classes):
    var = lambda mean: multivariate_normal.pdf(X , mean= means , cov= sigma)
    y_probs = np.apply_along_axis(var , 1 , means)*phi

    y_pred = np.argmax(y_probs , axis = 0)
    return classes[y_pred]