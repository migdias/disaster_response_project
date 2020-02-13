# Disaster Response Classification - Web App

The earth is an amazing planet that provides life in the most inhospitable places. However, not everything is perfect. All the time we are hit with constant messages about world disasters. With the rise of technology it is even easier to know what is really going on around the world. People and Organizations use tools like Twitter to offer or request help from others. So, how can we use these messages to make it easier to provide help to those in need? 

### 1. Summary of Project

Using **Natural Language Processing** (NLP) it is possible to analyse the messages sent and train a machine learning model to classify with type of disaster it was, if it is an offer of help or a request, and more. In this project I provide a simple machine learning algorithm based on word frequency to correctly classify and predict newer messages coming in Twitter.

Steps taken:
1. First the messages are processed, normalized, stripped and more
2. The words left, also known as **tokens**, are fit into a **Frequency Vectorizer** which get the frequency of words, which is then fed into a **TFIDF**.
3. These features are fed into a Machine Learning algorithm (in this case **AdaBoostClassifier**. Other models were used and tinkered with but this one seemed to provide the best results.).
4. Hyperparameter tuning using **GridSearchCV**.
5. Both the model and the data are ingested into a **web app** which provides a better visualization of the results.

### 2. Files

```bash
├── README.md
├── app
│   ├── run.py
│   └── templates
│       ├── go.html
│       └── master.html
├── data
│   ├── DisasterResponse.db
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   └── process_data.py
├── jupyter-notebooks
│   ├── DisasterResponse.db
│   ├── ETL\ Pipeline\ Preparation.ipynb
│   └── ML\ Pipeline\ Preparation.ipynb
└── models
    ├── classifier.pkl
    ├── train_classifier.py
    └── train_output.txt
```   

- The **data** folder contains the database with the processed data (obtained by running the **process_data.py**), the data divided into categories and messages in a csv format and a python script to process the data (both the messages and labels).

- The **models** folder contains a trained classifier that is obtained by running the **train_classifier.py**. It also has a **train_output.txt** which contains the printout of the model training, including best parameters.

- The **app** folder contains html templates for the app and a python script **run.py** that creates and deploys the web app.

- The **jupyter-notebooks** folder contains two jupyter notebooks that served as a support in writing the **process_data.py** and the **train_classifier.py**.

### 3. How to run

To run this web app, you can fork this repository or download it to your local machine and execute in the home folder:

```bash 
cd app
python run.py
```

If you want to rerun the processing and training before deploying the app please run the following commands:

- To run the ETL Pipeline that cleans data and stores in a database
```bash
python PROCESS_DATA_PYTHON_PATH MESSAGES_PATH CATEGORIES_PATH DATABASE_WRITE_PATH
```

- To run the ML Pipeline that trains the classifier and saves
```bash
python CLASSIFIER_PYTHON_PATH DATABASE_PATH MODEL_PATH
```

Example:

```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

#### 3.1. Dependencies

For this project you will need to have:
- pandas
- re
- nltk
- pickle
- scikit-learn
- plotly
- flask
- sqlalchemy

### 4. How does the Web App Works

After you run the run.py you can access the app by visiting http://0.0.0.0:3001/.

You stumble upon a page with 4 graphs containing a little bit of information regarding the training data.

You also have a text box to input a message. After you press classify you are routed too another page where this message will be classified into 36 categories. Try putting a message like "Please send help! Everything was destroyed by the earthquake and we have nowhere to live." See what the model classifies this into :)

