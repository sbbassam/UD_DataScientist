# import libraries
import re
import numpy as np
import pandas as pd
import sys
# workspace Session Control Class
import workspace_utils as ws
#SQL Database
from sqlalchemy import create_engine
# NLP
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#ML
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, make_scorer
#Pipeline, and features
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
# pickle to store model
import pickle

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    Categorize the verbs
    
    Input: Sentence to be categorized
    Output: verb tages
    '''
    
    def starting_verb(self, text):
        # tokenize by sentences
        sentence_list = nltk.sent_tokenize(text)
        # tokenize each sentence into words and tag part of speech
        for sentence in sentence_list:
            # index pos_tags to get the first word and part of speech tag
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            # return true if the first word is an appropriate verb or RT for retweet
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    """
    Load disaster dateset from SQLite & split dataset into messages and categories 
    
    Input: database_filepath: path of SQLite database file of processed dataset
    Return: X: Message part of disaster_message dataframe
            y: category part of disaster_message dataframe
            categories: names of categories, columns of y
    """
    # Load SQLite dataset
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)

    # split dataset and return message, categories, and column names
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = y.columns.tolist()

    return X, y, category_names


def tokenize(text):
    '''
    Clean and tokenize the message
    
    Input: text: the message for tokenization(X from load_data)
    Return: clean_tokens: token list of message
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():
    """
    Builds the pipeline of NLP + Classification, Tf-Idf as message 
    transformation and Random Foreset classifier + MultiOutputClassifier as 
    classification model
    
    Return: model: machine learning model described above
    """
    # pipeline containing Tf-Idf and Random Forest Classifier
    Tuned_MOC = MultiOutputClassifier(OneVsRestClassifier(LinearSVC(random_state=0, multi_class='crammer_singer',C = 5)))
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize,ngram_range=(1,5))),
                ('tfidf', TfidfTransformer(use_idf=False))
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', Tuned_MOC)
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance and print metrics of validation
    
    Input:model: classification model from build_model()
        X_test: test set of X(messages)
        Y_test: test set of y(categories)
        category_names: names of message categories(categories column names)
    """
    # print metrics of classification for each column
    Y_prediction = model.predict(X_test)
    print(classification_report(Y_test.values, Y_prediction, target_names=category_names))
    
    accuracy = (Y_prediction == Y_test).mean()
    print("Accuracy:\n", accuracy)
    


def save_model(model, model_filepath):
    """
    Save classification model to pickle file
    
    Input:model: validated classification model
        model_filepath: specified storage path
    """
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
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    with ws.active_session():
         main()