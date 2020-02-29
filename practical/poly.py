import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

m=100
X=6*np.random.randn(m,1)-3
y=0.5*X**2+X+2+np.random.randn(m,1)

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)
y_pred=lin_reg.predict(X)

from sklearn.metrics import mean_squared_error,r2_score
lie_mse1=mean_squared_error(y,y_pred)
line_rmse1=np.sqrt(lie_mse1)
print(line_rmse1)
r2_score(y,y_pred)

plt.title('Simple linear regression with FD')
plt.xlabel('random number')
plt.ylabel('value of theta')
plt.scatter(X,y)
plt.plot(X,y_pred)
plt.show()


m=100
X=6*np.random.randn(m,1)-3
y=0.5*X**2+X+2+np.random.randn(m,1)


from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=3,include_bias=False)
X_poly=poly.fit_transform(X)

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X_poly,y)
y_pred=lin_reg.predict(X_poly)


from sklearn.metrics import mean_squared_error,r2_score
lie_mse1=mean_squared_error(y,y_pred)
line_rmse1=np.sqrt(lie_mse1)
print(line_rmse1)
r2_score(y,y_pred)


plt.title('Simple linear regression with FD')
plt.xlabel('random number')
plt.ylabel('value of theta')
plt.scatter(X,y)
plt.plot(X,y_pred)
plt.show()


X_new=np.linspace(-10,10,100).reshape(-1,1)
X_new_poly=poly.fit_transform(X_new)
y_new=lin_reg.predict(X_new_poly)



plt.title('Simple linear regression with FD')
plt.xlabel('random number')
plt.ylabel('value of theta')
plt.scatter(X,y)
plt.plot(X,y_pred)
plt.show()




