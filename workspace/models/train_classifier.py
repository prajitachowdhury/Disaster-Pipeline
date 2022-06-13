import sys
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize


from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ExtraTreeClassifier
import pickle
import nltk
nltk.download('punkt')

def load_data(database_filepath):

    '''
    load data from the database
    
    Input: 
    database_filepath : database loc
 
    Returns:
    X: X feature dataframe
    Y: Y target dataframe for which we are predicting values
    cols: list of names of categories   
    '''
    engine = create_engine('sqlite:///%s' % database_filepath)
    print(engine)
    df = pd.read_sql_table('DisasterResponse.db',engine)
    
    df.drop(columns = ['original','genre'],inplace=True)
    df.fillna(0,inplace=True)
    cols = df.columns[2:]
    for col in cols:
        df[col].replace({0.0:int(0), 1.0:int(1),2.0:int(1)},inplace=True)
    
    X = df[['message']]

    Y = df.iloc[:, 2:]
    
    return X,Y,cols
    


def tokenize(text):
    '''
    tokenize text
    
    '''
    return word_tokenize(text)


def build_model():

	'''
	build model
	
	Input: None
	Returns: model built using GridSearchCV
	'''
    pipeline_cv = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            
        ])),

        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    parameters = {
        
       
        'clf__estimator__n_neighbors': [2]
        }
    cv = GridSearchCV(pipeline_cv, param_grid=parameters,verbose=3)
    
    return cv
    '''pipeline = Pipeline([
    ('vect',CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])'''
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
	'''
	evaluate the performance of the model
	
	Input:
	model : model used for training data
	X_test ,Y_test : test data
	category_names
	
	Returns:
	f1_score of the model
	'''
    y_pred = model.predict(X_test.values.flatten())
    test = y_pred.flatten()
    for col in Y_test.columns:
    	print("category: ", col)
    	classification_report(Y_test[col], Y_pred[col])
    score = f1_score(y_pred, Y_test,average='weighted')
    return score

def save_model(model, model_filepath):
	'''
	save model in pickle file
	
	Input: 
	model : to be saved
	model_filepath : loc of pickle file
	
	Returns: 
	None
	'''
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        X_train = X_train.values.flatten()
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
