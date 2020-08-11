from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class text_stats(BaseEstimator, TransformerMixin):    
    '''class returns texts statistics as a dataframe'''
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        length = len(X)
        unique_words = len(set(X))
        
        df = pd.concat([pd.Series(length), pd.Series(unique_words)], axis=1)
        df.columns = ['length','unique_words']
        return df