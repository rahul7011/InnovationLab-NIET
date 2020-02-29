#polynomial regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('housing.csv')
dataset.isnull().sum()

X = dataset.iloc[:, :-1].values    #only the independent one i.e,0 till n-1
y = dataset.iloc[:,9].values     #only the dependent i.e, the last one!

#np.unique(X[:,8])
#taking care of the missing data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(X[:,[4]])       #to avoid the dataleakage(testing vs train data)
X[:,[4]]=imputer.transform(X[:,[4]])
#pd.DataFrame(X[:,4]).isnull().sum()

#Encoding categorical data(categorized data)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X=LabelEncoder()
X[:,8]=labelencoder_X.fit_transform(X[:,8])

#dummy encoding
transformer = ColumnTransformer([('onehotencoder', OneHotEncoder(),[8])],remainder='passthrough')
X = np.array(transformer.fit_transform(X), dtype=np.float)

# Avoiding the Dummy Variable Trap
    #removed the OCEAN TAG
X = X[:,1:]
#Building the optional model using backward elimination
    #adding x(0)=1 manually for the hypothesis as statsmodel don't add this by default
import statsmodels.api as sm
X=np.append(arr=np.ones((16146,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
"""OLS is the ordinary least square model 
and ,endog is for the dependent variable
 and exog is for the optimal matrix"""
regressor_OLS.summary()

X_opt=X[:,[0,1,2,3,5,6,7,8,9,10,11,12]]

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,2,5,6,7,8,9,10,11,12]]

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X=X_opt
del(X_opt)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)



from sklearn.metrics import mean_squared_error,r2_score
lie_mse1=mean_squared_error(y_test,regressor.predict(X_test))
line_rmse1=np.sqrt(lie_mse1)
print(line_rmse1)
r2_score(y_test,regressor.predict(X_test))



plt.title('Simple linear regression with FD')
plt.xlabel('random number')
plt.ylabel('value of theta')
plt.scatter(X,y)
plt.plot(X,y_pred)
plt.show()
