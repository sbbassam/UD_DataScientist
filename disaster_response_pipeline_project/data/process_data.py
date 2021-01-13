# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads data to a data frame
    Inputs:
        messges_filepath - Path to the CSV file containing messages
        categories_filepath - Path to the CSV file containing categories
    Output:
        df - Combined data containing messages and categories
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
  # Combine/merge two CSV files into one dataframe
    df = pd.merge(messages,categories,on='id')
   
    return df 
    
def clean_data(df):
    """
    Preprocess the data before loading to the database
    Input:
        df - Dataframe to be cleaned
    Output:
        Clenaed data frame
   
   
  
    """
    
   # Step1 - Split categories into separate category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.str.slice(stop=-2)
    # rename the columns of `categories`
    categories.columns = category_colnames
    
  #Step2 - Convert category values to just numbers 0 or 1
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # replace 2 with 1 in related column 
    categories.related.replace(2,1,inplace=True)
  #Step3 - Replace categories column in df with new category columns
    # drop the original categories column from `df`
    df.drop(labels=['categories'], axis=1, inplace=True)  
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
  #Step6 - Remove duplicates
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df
    


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('Messages', engine, index=False,if_exists='replace')
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()