import numpy as np
import pandas as pd
# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

class LinearRegressionTuCode:
    def __init__(self, has_bias=True):
        self.has_bias = has_bias
        self.w = None
        self.coef_ = None
        self.intercept_ = None 

    def add_bias(self, X):
        if self.has_bias:
            ones = np.ones((X.shape[0], 1))
            X_with_bias = np.concatenate((ones, X), axis=1)
            return X_with_bias
        return X

    def fit(self, X, y):
        X_with_bias = self.add_bias(X)
        self.w = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y # w = np.linalg.pinv(X_train@X_train.T)@X_train@y_train
        self.intercept_ = self.w[0] if self.has_bias else 0
        self.coef_ = self.w[1:] if self.has_bias else self.w

    def predict(self, X):
        X_with_bias = self.add_bias(X)
        return X_with_bias @ self.w #f(x)=X_T@W

    
def NSE(y_test, y_pred):
    return 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))

data = pd.read_csv('Gold_Price.csv')

dt_train,dt_test = train_test_split(data,test_size=0.3,shuffle=True)

X_train = dt_train.drop(['Date','Price'], axis = 1) 
y_train = dt_train['Price'] 
X_test= dt_test.drop(['Date','Price'], axis = 1)
y_test= dt_test['Price']


reg = LinearRegressionTuCode()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print('R2 score: ', r2_score(y_test, y_pred))
print('NSE: ', NSE(y_test, y_pred))
print('MAE: ', mean_absolute_error(y_test, y_pred))
print('RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred)))