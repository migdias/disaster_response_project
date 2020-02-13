import sys

# SQL
from sqlalchemy import create_engine

# Data Processing
import pandas as pd
import re
import numpy as np

# Save Model
import pickle as pkl

# Text Processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Models
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# Metrics and Pipelines
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

stop_words = stopwords.words('english')

def get_score(model, test_y, pred_y, category_names):
    """ 
    Description: Prints the model accuracy and f1 metrics.
  
    This function compares the predictions against the actual labels and returns a classification report for each and every category label.

    Parameters: 
    model (obj): The NLP trained model
    test_y (df): The actual labels of the test set
    pred_y (df): The predicted labels of the test set
    category_name (list): A list with the category names
  
    Returns: 
    None
    
    Output:
    A printed documentation regarding the accuracy and other metrics of the model, along with the best parameters from the GridSearchCV
  
    """
    
    # Initialize metric lists
    f1 = []
    acc = []
    
    # For each category print classification report and append metrics to list
    for i in range(test_y.shape[1]):
        print('################   ' + category_names[i] + '   ##################')
        print(classification_report(test_y.iloc[:,i], pred_y[:,i]))
        f1.append(f1_score(test_y.iloc[:,i], pred_y[:,i], average='macro'))
        acc.append(accuracy_score(test_y.iloc[:,i], pred_y[:,i]))
        
    print('#######################################################################')
    print('#########################   FINAL RESULTS   ###########################')
    print('#######################################################################')
    
    # Print F1-Macro metric (because it is the best one to deal with unbalanced data
    # plus print the total accuracy and the best parameters of the model.
    print("The Mean score of F1-Macro is: {}".format(pd.Series(f1).mean()))
    print("The Mean accuracy score is: {}".format(pd.Series(acc).mean()))
    print("The Best Parameters for the model are: {}".format(model.best_params_))

def load_data(database_filepath):
    """ 
    Description: Loads the Disaster Response dataset.
  
    This function loads the cleaned dataset from a database and sets the features and labels, as well as the category names, accordingly.

    Parameters: 
    database_filepath (string): The path of the database file to load.
    
    Returns: 
    X (df): The features that are going to be used to train the model.
    Y (df): The labels that are going to be predicted (MultiOutput Labels).
    category_names (list): The names of each category to be predicted.
    """
    
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(database_filepath, con=engine)
    
    # Split into feature and label
    X = df["message"]
    Y = df[['related', 'request',
           'offer', 'aid_related', 'medical_help', 'medical_products',
           'search_and_rescue', 'security', 'military', 'child_alone', 'water',
           'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees',
           'death', 'other_aid', 'infrastructure_related', 'transport',
           'buildings', 'electricity', 'tools', 'hospitals', 'shops',
           'aid_centers', 'other_infrastructure', 'weather_related', 'floods',
           'storm', 'fire', 'earthquake', 'cold', 'other_weather',
           'direct_report']]
    
    # Get category names
    category_names = Y.columns.values
       
    return X, Y, category_names


def tokenize(text):
    """ 
    Description: Tokenizes a string of text.
  
    This function takes a string of text and executes and NLP processing job, normalizing, removing punctuation and stop words and other NLP tasks, resulting in single tokens.

    Parameters: 
    text (string): A single text string to be processed.
    
    Returns: 
    clean_tokens (list): A list of clean word tokens.
    """
    
    # 1. Normalize
    text = text.lower()
    
    # 2. Punctuation Removal
    text = re.sub(r"[^a-zA-Z0-0]", " ", text)
    
    # 3. Tokenization
    tokens = word_tokenize(text)
    
    # 4. Stopword Removal
    tokens_without_stop = [w for w in tokens if w not in stop_words]
    
    # 5. Lemmatizer and strip
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for w in tokens_without_stop:
        clean_tokens.append(lemmatizer.lemmatize(w).strip())
        
    return clean_tokens


def build_model():
    """ 
    Description: Builds an optimized machine learning model.
  
    This function builds a machine learning model using a ML pipeline and GridSearchCV to get an optimized text classification model.

    Parameters: 
    None
    
    Returns: 
    cv (obj): An optimized machine learning model.
    """
    
    # Create Pipeline 
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer(smooth_idf=False)),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    # Setting parameters for grid search
    parameters = {
        'clf__estimator__learning_rate': (0.1, 0.5, 1.0, 1.5),
        'clf__estimator__n_estimators': [50, 100, 150, 200],
        'vect__ngram_range': ((1, 1), (1, 2), (1,3)),
    }

    # Computing Grid Search
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, n_jobs=-1,verbose=1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """ 
    Description: Predicts and evaluates a model.
  
    This function predicts labels for a testset and compares it to the actual real labels. It then prints the evaluation onto the terminal.

    Parameters: 
    model (obj): The trained machine learning model.
    X_test (df): The test messages.
    Y_test (df): The labels of those messages.
    category_names (list): A list containing the category names.
    
    Returns: 
    None
    """
    
    # Getting Predictions
    y_pred = model.predict(X_test)
    
    # Print out Evaluation
    get_score(model, Y_test, y_pred, category_names)

def save_model(model, model_filepath):
    """ 
    Description: Saves a model.
  
    This function will save a machine learning model as a pickle file.

    Parameters: 
    model (obj): A trained machine learning model.
    model_filepath (strins): The path to which the model will be saved.
    
    Returns: 
    None
    """
    
    # Saving model
    pkl.dump(model, open(model_filepath, 'wb'))


def main():
    """ 
    Description: Trains and saves a machine learning model.
  
    This function will load and transform a dataset creating and saving a machine learning model.
    """
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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