from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class text_stats(BaseEstimator, TransformerMixin):    
    '''class returns texts statistics as a dataframe'''
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        length = X.apply(len)
        unique_words = X.apply(lambda x: len(set(w for w in x.split())))
        
        df = pd.concat([length, unique_words], axis=1)
        df.columns = ['length','unique_words']
        return df