import json
import plotly
import pandas as pd
import numpy as np
import itertools
from collections import Counter
import string

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')

from flask import Flask
from flask import render_template, request, jsonify
import plotly.graph_objs as goobj
from sklearn.externals import joblib
from sqlalchemy import create_engine

from custom_transformers import text_stats



app = Flask(__name__)

add_stopwords = ["'s", "n't", "u", "http", "also", '"', "..", "bit.ly"]

def tokenize(text):
    """
    This function tokenizes, lemmatizers, and lowercases text, removes stopwords and punctuation

    Arguments:
        x: string or Pandas series 

    Returns:
        cleaned tokens
    """
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.lower() not in stopwords.words("english") and t.lower() not in string.punctuation and t.lower() not in add_stopwords]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)
df['clean_message'] = df['message'].apply(lambda text: tokenize(text))
df_melt = pd.melt(df, id_vars=['clean_message'], value_vars = df.columns[4:40])
df_melt = df_melt[df_melt['value'] == 1]


# load model
model = joblib.load("../models/classifier.pkl")



def return_graphs():
    """
    This function returns three charts for the master page of the app
    """
 
    # data for the first chart

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # data for the second chart
   
    top15 = df_melt.groupby('variable').sum()['value'].reset_index().sort_values('value', ascending=False).head(15)
    
    cats = list(top15['variable'])
    cats = [word.replace('_', ' ').title() for word in cats]
    values = top15['value']
    
    # drawing the first and the second charts 
    
    graphs = [
        {
            'data': [
                goobj.Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                goobj.Bar(
                    x=cats,
                    y=values
                )
            ],

            'layout': {
                'title': 'Top-15 Categories of the Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # data for the third chart

    feats = df_melt.variable.unique().tolist()
    words_counts = pd.DataFrame()

    for feat in feats:
        df_melt_feat = df_melt[df_melt['variable'] == feat]
        all_words = list(itertools.chain(*df_melt_feat['clean_message']))
        counts = Counter(all_words)
        counts_df = pd.DataFrame(counts.most_common(15),
                                columns=['words', 'counts'])
        counts_df['category'] = np.repeat(feat, 15)
        words_counts = words_counts.append(counts_df)
    
    graph_three = []
    
    for feat in feats:
        x_val = words_counts[words_counts['category'] == feat].words.tolist()
        y_val = words_counts[words_counts['category'] == feat].counts.tolist()
        graph_three.append(
                goobj.Bar(
                    x = x_val,
                    y = y_val,
                    visible=feat == 'related',
                    name = feat
                    )
                 )
                                  
    buttons = []
    
    for i, name in enumerate(feats):
        visible = [False]*len(feats)
        visible[i] = True
        buttons.append(
            dict(
                method='update',
                args=[{'visible': visible}],
                label=name
            ))
    
    layout_three = dict(
        title='The most frequent words',
        yaxis=dict(title='Counts'),
        updatemenus=list([
            dict(
                x=-0.1,
                y=1,
                buttons=buttons,
                yanchor='top'
            )
        ]),
    )
    
    graphs.append(dict(data=graph_three, layout=layout_three))
    
    return graphs

@app.route('/')
@app.route('/master')
def master():
    
    graphs = return_graphs()
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    
    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()