import matplotlib.pyplot as plt
import numpy as np
from model_functions import fit_2d_polynomial, R2
from scraper import scrape

#fetch data again--this time, we're looking at life expectancy as a function of GDP
data=scrape()
X=data["GDP Per Capita(usd)"]

#apply logarithmic transformation to X data
X=[np.log(i) for i in X]
Y=data["Total"]

#plot transformed x against Y
plt.scatter(X, Y, c="orange")

#coefficients in the quadratic formula (i.e. the 'a','b',and'c' in the equation Y = a*(x**2) + b*(x) = c
x= fit_2d_polynomial(X,Y)[0]
y_hat = fit_2d_polynomial(X,Y)[1]
Y = fit_2d_polynomial(X,Y)[2]

#plot and calculate R^2 for model
plt.plot(x,y_hat,c='red')
plt.xlabel("Natural Log of GDP Per Capita (USD)")
plt.ylabel("Life Expectancy (years)")
plt.show()
print(f"R2: {R2(Y,y_hat)}")