""" Auxiliary functions for the CESAMO Encoder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.preprocessing import MinMaxScaler 
except: 
    raise Exception('Scikit-Learn 0.22.2+  not available')
def replace_in_df(df, mapping):
    """ Replaces categories by numbers according to the mapping
        If a category is not in mapping, it gets a random code
        mapping: dictionary from categories to codes
    """
    # Ensure df has the right type
    if not(isinstance(df,(pd.DataFrame))):
        try:
            df = pd.DataFrame(df)
        except:
            raise Exception('Cannot convert to pandas.DataFrame')

    cat_cols = categorical_cols(df)

    # Updates the mapping with random codes for categories not 
    # previously in the mapping
    for x in cat_cols:
        values = np.unique(df[x])
        for v in values:
            if not(v in mapping[x]):
                mapping[x][v] = np.random.uniform(0,1)

    return df.replace(mapping)

def scale_df(df):
    """ Scale all numerical variables to [0,1]
    """
    numerical_cols = [x for x in list(df.columns) if x not in categorical_cols(df)]
    sc = MinMaxScaler()

    for x in numerical_cols:
        if min(df[x].values) < 0.0 or 1.0 < max(df[x].values):
            df.loc[0:, x] = sc.fit_transform(df[x].values.reshape(-1,1))

    return df


def is_categorical(series):
    """ Tests if the column of a dataframe is categorical
    """
    return series.dtype.name == 'category' or series.dtype.name == 'object'

def categorical_cols(df): 
    """ Return the column numbers of the categorical variables in df
    """
    cols = []
    # Rename columns as numbers
    df.columns = range(len(df.columns))
    
    for x in df.columns: 
        if is_categorical(df[x]):
            cols.append(x)
    return cols

def categorical_instances(df):
     """ Returns an array with all the categorical instances in df, 
         column by column
     """
     instances = {}
     cols = categorical_cols(df)
     for x in cols:
         instances[x] = list(np.unique(df[x]))
 
     return instances

def num_categorical_instances(df):
    """ Returns the total number of categorical instances in df
    """
    instances = categorical_instances(df)

    return np.sum([len(instances[x]) for x in instances])


def set_categories(df, cat_cols=[]):
    already_categorical = categorical_cols(df)
    cols = [x for x in cat_cols if x not in already_categorical]
    categories = {}
    for x in cols:
        unique = np.unique(df[x])
        xcats = {}
        for v in unique:
            xcats[v]= 'X'+str(x)+'_'+str(v)

        categories[x] = xcats
    return df.replace(categories)
