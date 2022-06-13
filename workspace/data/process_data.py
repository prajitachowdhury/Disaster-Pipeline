import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
	'''
	load data from csv files and merge to single pandas dataframe
	
	Input: 
	messages_filepath - filepath to messages_csv file
	categories_filepath - filepath to categories_csv file
	
	Returns:
	df: dataframe merging categories and merges
	'''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df
    
def clean_data(df):
    '''
    clean dataframe to transorm into usable format
    
    Input:
    df: Dataframe to be worked upon
    
    Returns:
    df: Transformed Dataframe 
    '''
    categories = df['categories'].str.split(";",expand=True)
   
    row = categories.loc[0,:]
    s = slice(-2)
    category_colnames = list()
    for val in row:
        category_colnames.append(val[s])
    
    
    categories.columns = category_colnames
   
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(-1)
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df.drop(columns=['categories'],inplace=True)
    
    df=pd.concat([df,categories],axis=1)
    
   # check number of duplicates
    df.duplicated().sum()
    
   # drop duplicates
    df.drop_duplicates(inplace=True)
    
   # check number of duplicates
    df.duplicated().sum()
    return df 
    
def save_data(df, database_filename):
    '''
    save dataframe into database
    
    Input: 
    df - Dataframe to be saved
    database_filename - name of database
    
    Returns:
    None
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse.db', engine, index=False,if_exists='replace')
    print(engine)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
      
        print('Cleaning data...')
       
        df = clean_data(df)
        
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, "DisasterResponse.db")
        
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
