import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import numpy as np
import pickle
from collections import defaultdict

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from custom_transformers import text_stats


def load_data(database_filepath):
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', engine)  
    X = df.iloc[:,0:4]
    Y = df.iloc[:,5:40]
    Y = Y.drop('child_alone', axis = 1) # remove this variable as it has only 0s
    category_names = list(Y.columns)
    
    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.lower() not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    model = Pipeline([
        ('features', FeatureUnion([
        
            ('nlp_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        
            ('text_stats', text_stats())
        ])),
    
        ('clf', MultiOutputClassifier(RandomForestClassifier(min_samples_split = 4, n_estimators = 300)))
    ])
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred = model.predict(X_test)
    
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = category_names
    
    for i in category_names:
        print(classification_report(Y_test[i], y_pred[i]))


def save_model(model, model_filepath):
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train['message'], Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test['message'], Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()