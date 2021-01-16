import numpy as np
import matplotlib.pyplot as plt
from model_functions import linear_log, R2
from scraper import scrape
from linearregression import LinearRegression
import scipy

#fetch data by calling function in webscraper module
data=scrape()

#GDP is independent variable,('total')Life expectancy is dependent variable
X=np.array(data["GDP Per Capita(usd)"])
Y=data["Total"]

#plug transformed X values and Y values into formula for simple linear regression
model=LinearRegression(X,Y)

"""
logarithmic transformation is applied to normalize X data
transformed X values are plotted against Y values
"""

"""
normality testing using both 
I.) a histogram plot
II.) a Q-Q plot
--------
"""

scipy.stats.probplot([np.log(i) for i in X], dist="norm", plot=plt)
plt.show()
plt.hist([np.log(i) for i in X])
plt.show()

"""
These data appear to loosely approximate a normal distribution--
since log transformations assume a normal distribution, and the data don't strictly follow one, 
the model won't be perfect.
"""

plt.scatter([np.log(i) for i in X], Y, c="orange")
plt.xlabel("Natural Log of GDP Per Capita (USD)")
plt.ylabel("Life Expectancy (years)")

#sort X and Y values and plot regression line
x_vals,y_pred,Y = linear_log([np.log(i) for i in X],Y)
plt.plot(x_vals, y_pred, c="red")
plt.show()


#get R2 value for model
print(f"R2: {R2(Y,y_pred)}")

#use model to predict unseen data
print(np.log(100000))
print(model.predict(np.log(100000)))