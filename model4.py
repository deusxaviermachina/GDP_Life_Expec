from model_functions import fit_2d_polynomial, R2
from scraper import scrape
import matplotlib.pyplot as plt
import numpy as np

#fetch data
data=scrape()

#GDP is dependent variable,('total')Life expectancy is independent variable
X=data["Total"]
Y=data["GDP Per Capita(usd)"]

#apply natural log transformation to Y data
Y=[np.log(i) for i in Y]

#fit to second order polynomial, plot X data points against transformed Y data points
model=np.polyfit(X,Y,2)
plt.scatter(X,Y,color='black')

#fit quadratic curve (of the form a(x**2)+b(x)+c to data
x=fit_2d_polynomial(X,Y)[0]
y=fit_2d_polynomial(X,Y)[1]
Y=fit_2d_polynomial(X,Y)[2]

#plot data and label axes
plt.plot(x,y, c='magenta')
plt.xlabel("Life Expectancy (years)")
plt.ylabel("Natural Log of GDP Per Capita (USD)")
plt.show()

#get correlation coefficient/R2 for model
print(f"R2: {R2(Y,y)}")

#testing the model
e=2.718281828459045235
x=75
print(model[0]*(x**2) + x*model[1] + model[2], e ** model[0]*(x**2) + x*model[1] + model[2])
