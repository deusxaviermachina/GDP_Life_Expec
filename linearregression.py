class LinearRegression:
    """
    simple linear regression, built from scratch
    """
    def __init__(self, independent, dependent):
        self.independent=independent
        self.dependent=dependent
        assert len(self.independent) == len(self.dependent)

    def fit_data(self):
        X=self.independent
        y=self.dependent
        N = len(X)
        x_sq = sum([i**2 for i in X])
        numerator1=(x_sq*sum(y)) - (sum([i*j for i,j in zip(X,y)]) * sum(X))
        numerator2= (N*(sum([i*j for i,j in zip(X,y)]))) - sum(X)*sum(y)
        denominator=(N*x_sq) - sum(X)**2
        B=numerator1/denominator
        M=numerator2/denominator
        y_pred = [(M*i)+B for i in X]
        return M, B, y_pred

    def predict(self, x):
        """
        use model to make predictions
        :param x:
        :return:
        """
        M,B,y_pred = self.fit_data()
        return M*x +B