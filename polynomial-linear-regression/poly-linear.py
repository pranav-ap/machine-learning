# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset  = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # independent vars matrix
y = dataset.iloc[:, 2].values # dependent var vector

## feature scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test) # fit not needed

# fit simple linear regressor to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression() # takes care of feature scaling for us
lin_reg.fit(X, y)

# fit polynomial regressor to dataset
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 4)
X_poly = poly.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# visualise slr
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or bluff (Linear regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# visualise plr
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1)) # from vector to matrix
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or bluff (Polynomial regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# predict using slr
lin_reg.predict(6.5)

# predict using polynomial regression
lin_reg_2.predict(poly.fit_transform(6.5))

