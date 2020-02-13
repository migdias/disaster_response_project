import sys
import pandas as pd
import sqlalchemy

def load_data(messages_filepath, categories_filepath):
    """ 
    This function loads messages and category data and 
    merges them together for later cleaning.
    
    Parameters: 
    messages_filepath (string): File path of the messages.csv data file.
    categories_filepath (string): File path of the catefories.csv data file.
  
    Returns: 
    df (pandas dataframe): A pandas dataframe of the whole data. 
    """
    
    # Load messages and categories files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge both into a single dataframe
    df = messages.merge(categories, on='id')
    
    return df

def clean_data(df):
    """ 
    This function cleans and processes the message and category data.
    
    Parameters: 
    df (pandas dataframe): The pandas dataframe of the merged categories and messages data.
  
    Returns: 
    df (pandas dataframe): A cleaned and processed pandas dataframe
    """
    
    # Split category names and value
    categories = df.categories.str.split(pat=';', expand=True)

    # Get category column names and renaming the columns of the dataframe
    category_colnames = categories.iloc[0].str.split(pat='-').str[0]
    categories.columns = category_colnames
    
    # Get the values for each category column
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split(pat='-').str[1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Drop the first unprocessed categories column
    df.drop('categories', axis=1,inplace=True)
    
    # Concat the dataframe with the newly processed categories
    df = pd.concat([df, categories], axis=1)

    # Drop duplicates in the data
    df = df.drop_duplicates()
    
    # Change to related to binary
    df.related.replace(2, 1, inplace=True)
    
    return df

    
def save_data(df, database_filename):
    """ 
    This function saves the data into a SQL database.
    
    Parameters: 
    df (pandas dataframe): The pandas dataframe of the cleaned and processed data.
    database_filename (string): The name of the database to create wth the data
  
    Returns: 
    SQL Database
    """
    
    # Create SQL engine
    engine = sqlalchemy.create_engine('sqlite:///{}'.format(database_filename))
    
    # Write to an SQL database
    df.to_sql(database_filename, engine, index=False)


def main():
    """ 
    This is the main of the script. It runs all the functions above with some logging 
    and error handling.
    """
    
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