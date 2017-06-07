# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset  = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values # independent vars vector
y = dataset.iloc[:, -1].values # dependent var vector

# encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3]) # dummy encoding
X = onehotencoder.fit_transform(X).toarray()

# avoiding dummy variable trap
X = X[:, 1:]  

# splitting dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# fit multiple linear regressor to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predict from test set
y_pred = regressor.predict(X_test)

# build optimal model using backward elimination
import statsmodels.formula.api as sm
## add a column of ones 
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

## FYI
# lower the p-value the more significant the variable is 
# wrt independent variable 

## start from all predictors
X_opt = X[:, [0,1,2,3,4,5]]
## fit the model 
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() # FYI default SL = 5% = 0.05

## PASS 2
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

## PASS 3
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() 

## PASS 4
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

## PASS 5
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
