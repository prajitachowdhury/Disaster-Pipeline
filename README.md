## Disaster Messages - Classification 
#### Objective of project: 
To analyze disaster data to build a model that classifies disaster messages.

#### File Desriptions:
1. process_data.py : Python file to run the ETL Process on the source data 
2. train_simplifier.py: Python file to build and run Machine Learning pipeline model on the data loaded through previous file.
3. run.py: Python file to run the web application that allows you to enter a message that gets appropriately classified.

#### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

