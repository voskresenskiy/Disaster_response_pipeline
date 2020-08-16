# Disaster response pipeline

## Project motivation 

The current project is done in the framework of the [Data Scientist Nanodegree at Udacity](https://www.udacity.com/course/data-scientist-nanodegree--nd025) and includes two main goals:
- to develop a pipeline that handles disaster data provided by [Figure Eight](https://appen.com/) and builds multioutput classifier model predicting the category of a disaster message;
- to make an application that allows an user to predict a category for any new message. 

## Description of files

- **ETL Pipeline Preparation** notebook contains all the steps needed for data cleaning, handling and saving the final dataset into SQL database. 
- **ML Pipeline Preparation** notebook contains all my experiments with different classifiers
- The folder **data** includes disaster messages (*disaster_messages.scv*), categories for these messages (*disaster_categories.csv*), the final database with cleaned and preprocessed dataset (*DisasterResponse.db*), and the script cleaning, merging, and saving data (*process_data.py*) (the script is made on the base of ETL Pipeline Preparation notebook)
- The folder **models** includes the customer transformer calculating some statistics for an input text (*custom_transformers.py*) and the script with the pipeline for the final model (*train_classifier.py*).
- The folder **app** includes templates and script for launching the application.

## How to run the pipelines and the app

1) To run ETL pipeline that cleans data and stores in database:
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

2) To run ML pipeline that trains classifier and saves:
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3) Run the following command in the app's directory to run your web app:
`python run.py`

4) Go to http://0.0.0.0:3001/

## Libraries

To successfully reproduce my script on your machine, the following packages should be installed: pandas, nltk, numpy, sklearn, json, plotly, flask, sqlalchemy
