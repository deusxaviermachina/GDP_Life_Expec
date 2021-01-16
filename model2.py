import matplotlib.pyplot as plt
import numpy as np
from model_functions import linear_log, R2, normality_test, residual_plot
from scraper import scrape
from linearregression import LinearRegression
import pandas as pd

#fetch dataset by calling function from webscraping module
data=scrape()

#look at data
pd.plotting.scatter_matrix(data)
plt.show()

#GDP as a function of life expectancy--here, GDP is our dependent varible and life expectancy is our independent variable
X = data["Total"]
Y = data["GDP Per Capita(usd)"]

#apply log linear transformation and plug transformed X and Y values into formula for simple linear regression
Y=[np.log(i) for i in Y]
model=LinearRegression(X,Y)
plt.scatter(X,Y, c='purple')
x = linear_log(X,Y)[0]
y_pred = linear_log(X,Y)[1]
Y=linear_log(X,Y)[2]
plt.plot(x, y_pred,c='orange')
plt.xlabel("Life Expectancy (years)")
plt.ylabel("Natural Log of GDP Per Capita (USD)")
plt.show()

#get R^2 for model
print(f"R2: {R2(Y,y_pred)}")
x=75
e=2.718281828459045235

#is there autocorrelation of residuals?
residual_plot(X, y_pred, Y)
plt.show()

#how closely do data approximate a normal distribution?
normality_test(X)
plt.show()

