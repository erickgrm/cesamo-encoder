""" Implementation of the CESAMO Encoder. For details see 
    "On the encoding of categorical variables for machine learning applications", Chapter 3

    @author github.com/erickgrm
"""
# Libraries providing estimators
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.svm import SVR
from .polynomial_regression import PolynomialRegression, OddDegPolynomialRegression
from sklearn.neural_network import MLPRegressor

dict_estimators = {}
dict_estimators['LinearRegression'] = LinearRegression()
dict_estimators['SGDRegressor'] = SGDRegressor(loss='squared_loss')
dict_estimators['SVR'] = SVR()
dict_estimators['PolynomialRegression'] = PolynomialRegression(max_degree=3)
dict_estimators['Perceptron'] = MLPRegressor(max_iter=150, hidden_layer_sizes=(10,5))
dict_estimators['CESAMORegression'] = OddDegPolynomialRegression(max_degree=11)

from .utilities import *
from .encoder import *
import numpy as np
from scipy.stats import shapiro, normaltest
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
colours = ['blue', 'yellow']

class CESAMOEncoder(Encoder):

    def __init__(self, estimator_name='CESAMORegression', plot=False,
                 max_sampling=800):
        """ Allows any of the keys in dict_estimators as estimator_name
        """
        super(CESAMOEncoder, self).__init__()
        self.estimator = dict_estimators[estimator_name]
        self.num_predictors = 1
        self.codes = {}
        self.plot_flag = plot
        self.df = None
        self.categories = {}
        self.max_sampling = max_sampling
        self.categorical_var_list = []

    def fit(self, df, y=None, cat_cols=[]):
        if not(isinstance(df,(pd.DataFrame))):
            try:
                df = pd.DataFrame(df)
            except:
                raise Exception("Cannot convert to pandas.DataFrame")
        self.codes = {} # Restart in case the same instance is called again

        # Set which variables will be encoded
        self.categorical_var_list = var_types(df, cat_cols)
        # Scale and transform vars in cat_cols to be categorical if needed
        self.df = set_categories(scale_df(df.copy()), self.categorical_var_list)

        self.categories = categorical_instances(self.df)
        
        # Find codes variable by variable
        for x in self.categories:
            self.codes[x] = self.findcodes(x, self.estimator, self.num_predictors)

        # Try to free up memory
        del self.df
        

    def findcodes(self, col_num, estimator, num_predictors):
        # Ensure y has the correct type
        y = self.df[col_num].copy() # pandas.Series object
        X = self.df.copy().drop(self.df.columns[col_num], axis=1)

        sampled = {}
        normality = False
        while not(normality):
            # Propose new set of codes for y
            ycodes = np.random.uniform(0, 1, len(np.unique(y)))
            yenc = y.copy().replace(dict(zip(np.unique(y), ycodes)))
            
            # Choose secondary var randomly; if categorical, encode with random numbers 
            S = X[np.random.choice(X.columns)].copy()
            if is_categorical(S):
                codes = np.random.uniform(0, 1, len(np.unique(S)))
                S =  S.replace(dict(zip(np.unique(S), codes)))

            # Fit the estimator and get error 
            estimator.fit(S, yenc)
            error = mean_squared_error(yenc, estimator.predict(S))
            
            # Save error and codes
            sampled[error] = ycodes
            
            # Update normality flag; at least 19 samples are required by the 
            # D'angostino-Pearson normality test
            if len(sampled) > 19:
                normality = self.normal_test(list(sampled.keys())) or \
                            self.max_sampling <= len(sampled)
                
        # Plot the distribution of errors if plotting flag=True
        if self.plot_flag:
            if len(sampled) == self.max_sampling:
                print('max number of codes ('+str(self.max_sampling)+
                      ') sampled for variable', col_num)
            else: 
                print(len(sampled), ' codes sampled for  variable', col_num)
            self.plot_errors(list(sampled.keys()), col_num)

        # Choose codes corresponding to minimum error and replace in df
        best = {}
        best[col_num] =  dict(zip(np.unique(y), sampled[min(sampled, key=lambda k:k)]))
        self.df.replace(best)

        return  best[col_num]

    def normal_test(self, observations, alpha=0.05):
        # Perform D’Agostino and Pearson’s normality test
        stat, p = normaltest(observations)

        #stat, p = shapiro(observations) # Another possible test

        if p > alpha:
            return True  # The observations come from a normal variable
        else:
            return False # The observations do not come from a normal variable

    def plot_errors(self, errors, col_num):
        sns.set_style("whitegrid")
        plt.figure(figsize=(6,4))
        sns.distplot(errors, label='Variable '+str(col_num))
        plt.legend()
        plt.show()
