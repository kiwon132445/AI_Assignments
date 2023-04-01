import pandas as pd
import numpy as np
import pickle
import os.path
import torch
import time
import string

# import gensim
# from gensim.models import KeyedVectors

# import preprocessor as p
import spacy
from spacy.lang.en import stop_words
from spacy.pipeline.lemmatizer import Lemmatizer
from spacy.language import Language
from spacy_transformers import Transformer
from spacy.tokens import DocBin

# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer

# Initialization method
class Yelp_Manager:
    # ------------------------------------------------------------------------------------#
    # Initialized Variables
    text = []
    x = None
    # columns
    # review_id user_id business_id stars useful funny cool text date
    input_columns = ['text']
    irrelevant_columns = []
    output_columns = ['stars', 'useful', 'funny', 'cool']
    
    word_embedded_dataset_path_json = './Save/yelp_academic_dataset_review(word_embedded).json'
    word_embedded_dataset_path_spacy = './Save/yelp_academic_dataset_review(word_embedded).spacy'

    # Uninitialized Variables
    #     filepath
    #     model_name

    #     yelp

    #     x, y
    #     X_train, y_train
    #     X_valid, y_valid
    #     X_test, y_test
    # ------------------------------------------------------------------------------------#

    def __init__(self, filepath='./yelp_dataset/yelp_academic_dataset_review.json', model="", experiment="", nrows=0, preprocessed=True):
        self.yelp = None
        self.filepath = filepath
        self.model = model
        self.experiment = experiment
        self.nrows = nrows
        self.preprocessed = preprocessed

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nlp = spacy.load("en_core_web_lg")
        self.doc_bin = DocBin()
        # self.nlp = spacy.load("en_core_web_trf")
        # self.transformer = pipeline("sentiment-analysis", model="roberta-base")
    
    # Loads the file and converts content into useable datatype
    def initialize(self):
        self.load_json()
        #self.load_doc_bin()
        self.convert_yelp()
    
    # Preprocess the data if it wasn't already, split the x and y
    def preprocess_dataset(self):
        self.x_y_split()
        if not self.preprocessed:
            self.spacy_word_embedding(self.x)
            self.save_embedded_json(self.x)
            #self.save_doc_bin()
        
    # Split the data into training, validation, and test sets
    def process_dataset(self):
        self.train_valid_test_split()
    
    # Converts dataframe content and checks for irrelevant columns
    def convert_yelp(self):
        self.yelp['text'] = self.yelp['text'].astype('str')

        for columns in self.yelp.columns:
            if columns == 'review_id':
                self.irrelevant_columns.append('review_id')
            elif columns == 'user_id':
                self.irrelevant_columns.append('user_id')
            elif columns == 'business_id':
                self.irrelevant_columns.append('business_id')
            elif columns == 'date':
                self.irrelevant_columns.append('date')

        self.yelp = self.yelp.drop(columns=self.irrelevant_columns)
    
    # Splits dataframe into x and y sets
    def x_y_split(self):
        if self.x is None:
            self.x = self.yelp.drop(columns=self.output_columns)
        self.y = self.yelp.drop(columns=self.input_columns)
    
    # Split the data into training, validation, and test sets
    def train_valid_test_split(self):
        self.X_train, X_remaining, self.y_train, y_remaining = train_test_split(self.x, self.y, train_size=0.8)
        self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(X_remaining, y_remaining, test_size=0.5)
    
    # Count vectorizer
    def count_vectorize(self, dataset):
        vec = CountVectorizer()
        return vec.fit_transform(dataset)

    # ------------------------------------------------------------------------------------#
    # Word Embedding
    def spacy_word_embedding(self, dataset):
        count = 0
        increments = 10000
        checkpoint = increments
        remaining = len(dataset['text'])
        starttime = time.time()
        
        for doc in self.nlp.pipe(dataset['text'].tolist(), n_process=10, disable=['tok2vec', 'parser', 'senter', 'ner']):
            clean_text = self.clean_text(doc)
            # self.doc_bin.add(self.nlp(clean_text))
            self.text.append(clean_text)
            count += 1
            if count == checkpoint:
                endtime = time.time()
                print("Remaining preprocess data:", remaining, " Time past:", endtime-starttime, end='\r')
                checkpoint += increments
                remaining -= increments
                
        self.x['text'] = self.text
    
    def clean_text(self, doc):
        lowercase = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in doc]
        lemmatize = [word for word in lowercase if word not in stop_words.STOP_WORDS and word not in string.punctuation]
        text = " ".join(lemmatize)
        return text
    # ------------------------------------------------------------------------------------#
    # Models
    
    
    # ------------------------------------------------------------------------------------#
    # Techniques
    

    #############################################################################################
    # Saving and loading data
    
    # Save model
    def save_model(self, model):
        filename = "./Save/" + self.model + "_" + self.technique + ".sav"
        pickle.dump(model, open(filename, 'wb'))
    
    # Load model
    def load_model(self, path):
        filename = path
        if os.path.isfile(filename):
            return pickle.load(open(filename, 'rb'))
        else:
            print("(" + path + ")" + "No such file exists, starting new model")
    
    # Save dataset from json
    def load_json(self):
        if self.filepath == './yelp_dataset/yelp_academic_dataset_review.json':
            if os.path.isfile(self.word_embedded_dataset_path_json) and self.preprocessed:
                if self.nrows == 0:
                    self.x = pd.read_json(self.word_embedded_dataset_path_json, lines=True)
                    self.x_backup = self.x
                else:
                    self.x = pd.read_json(self.word_embedded_dataset_path_json, lines=True, nrows=self.nrows)
                    self.x_backup = self.x
            if self.nrows == 0:
                self.yelp = pd.read_json(self.filepath, lines=True)
            else:
                self.yelp = pd.read_json(self.filepath, lines=True, nrows=self.nrows)
        else:
            if os.path.isfile(self.filepath):
                if self.nrows == 0:
                    self.yelp = pd.read_json(self.filepath, lines=True)
                else:
                    self.yelp = pd.read_json(self.filepath, lines=True, nrows=self.nrows)
            else:
                print("No such file exist")
    
    # Save word embedded dataset
    def save_embedded_json(self, dataset):
        dataset.to_json(self.word_embedded_dataset_path_json, orient="records", lines=True)
        
#     def save_doc_bin(self):
#         self.doc_bin.to_disk(self.word_embedded_dataset_path_spacy)
    
#     def load_doc_bin(self):
#         if os.path.isfile(self.word_embedded_dataset_path_spacy) and self.preprocessed:
#             self.doc_bin = DocBin().from_disk(self.word_embedded_dataset_path_spacy)
    #############################################################################################
