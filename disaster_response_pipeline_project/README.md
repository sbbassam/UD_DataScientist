# Disaster Response Pipeline Project

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)
5. [Instructions](#instructions)

### Installations<a name="installation"></a>
This project uses common libraries in Anaconda, and following packges are needed for NLP part of the project:

- punkt
- wordnet
- averaged_perceptron_tagger

### Project Motivation<a name="motivation"></a>
This project is about categorizing the disaster massages, so that they can be sent to an appropriate disaster relief agency. The data used for this project is provided by figure eight and contains real messages that were sent during disaster events. It contains a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

### File Descriptions<a name="files"></a>
There are three components for this project.

#### 1. ETL Pipeline
This part of the project is the process of Extract, Transform, and Load. The "process_data.py", reads the dataset, clean the data, and then store it in a SQLite database. Following are the steps:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

#### 2. ML Pipeline
The machine learning pipeline "ML Pipeline Preparation.ipynb" uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). The final model is used in the train_classifier.py.
The ML pipeline steps are:

- Load data from the SQLite database
- Split the dataset into training and test sets
- Build a text processing and machine learning pipeline
- Train a model using GridSearchCV
- Output results on the test set
- Export the final model as a pickle file

#### 3. Flask Web App
This step, dispays the results in a Flask web app. You will need to upload your database file and pkl file with your model. The template files for the simple visualization is provided in a../template directory, for home page(master.html) and result page (go.html). Following is snapshot of the pages:

![HomePage](https://raw.githubusercontent.com/sbbassam/UD_DataScientist/master/disaster_response_pipeline_project/app/templates/HomePage.png)

![ResultPage](https://raw.githubusercontent.com/sbbassam/UD_DataScientist/master/disaster_response_pipeline_project/app/templates/ResultPage.png)



### Licensing, Authors, Acknowledgements, etc.<a name="licensing"></a>
Credits must be given to Udacity for the starter codes and FigureEight for provding the data used by this project.

### Instructions:<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
