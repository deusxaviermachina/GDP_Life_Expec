import numpy as np
from linearregression import LinearRegression
import scipy.stats as stats
import matplotlib.pyplot as plt

def linear_log(X,Y):
    m, b, y_pred = LinearRegression(X, Y).fit_data()
    ordered_pairs = sorted([(i, j) for i, j in zip(X, y_pred)])
    x_vals = [i[0] for i in ordered_pairs]
    y_pred = [i[1] for i in ordered_pairs]
    Y = [j for (i, j) in sorted([(i, j) for i, j in zip(X, Y)])]
    return x_vals,y_pred,Y

def fit_2d_polynomial(X,Y):
    a,b,c = np.polyfit(X, Y, 2)
    x_quadratic = [a*(i**2) + b*i + c for i in X]
    ordered_pairs = sorted([(i, j) for i, j in zip(X, x_quadratic)])
    inputs = [i[0] for i in ordered_pairs]
    outputs = [i[1] for i in ordered_pairs]
    Y = [j for (i, j) in sorted([(i, j) for i, j in zip(X, Y)])]
    return inputs,outputs,Y

def R2(Y,y_predict):
    ss_resid=sum([(i-j)**2 for i,j in zip(Y,y_predict)])
    ss_tot=sum([(i-j)**2 for i,j in zip(Y,[np.mean(Y) for i in Y])])
    r2=1-(ss_resid/ss_tot)
    return r2

def normality_test(X):
    # does data follow a normal distribution?
    quantiles_plot= stats.probplot(X, dist="norm", plot=plt)
    return quantiles_plot

def residual_plot(X, predicted, observed):
    resid_plot = plt.scatter(X, [i - j for i, j in zip(predicted, observed)])
    plt.title("residual plot")
    plt.xlabel("independent variable")
    plt.ylabel("residuals")
    return resid_plot
