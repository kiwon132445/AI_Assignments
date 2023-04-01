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
    input_columns = ['text']
    irrelevant_columns = []
    output_columns = ['stars', 'useful', 'funny', 'cool']
    
    word_embedded_dataset_path = './Save/yelp_academic_dataset_review(word_embedded_only).json'

    # Uninitialized Variables
    #     filepath
    #     model_name

    #     yelp

    #     x, y
    #     X_train, y_train
    #     X_valid, y_valid
    #     X_test, y_test
    # ------------------------------------------------------------------------------------#

    def __init__(self, filepath='./yelp_dataset/yelp_academic_dataset_review.json', model_name="", nrows=0, preprocessed=True):
        self.yelp = None
        self.filepath = filepath
        self.model_name = model_name
        self.nrows = nrows
        self.preprocessed = preprocessed

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nlp = spacy.load("en_core_web_lg")
        # self.nlp = spacy.load("en_core_web_trf")
        # self.transformer = pipeline("sentiment-analysis", model="roberta-base")

    def initialize(self):
        self.load_json()
        self.convert_yelp()

    def preprocess_dataset(self):
        if not self.preprocessed:
            self.yelp['text'], self.yelp['vectors'] = self.spacy_word_embedding(self.yelp)
            self.save_embedded_json(self.yelp)

    def process_dataset(self):
        self.x_y_split()
        self.train_valid_test_split()

    # columns
    # review_id user_id business_id stars useful funny cool text date
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
            elif columns == 'vectors':
                self.input_columns.append('vectors')

        self.yelp = self.yelp.drop(columns=self.irrelevant_columns)

    def x_y_split(self):
        self.x = self.yelp.drop(columns=self.output_columns)
        self.y = self.yelp.drop(columns=self.input_columns)

    def train_valid_test_split(self):
        self.X_train, X_remaining, self.y_train, y_remaining = train_test_split(self.x, self.y, train_size=0.8)
        self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(X_remaining, y_remaining, test_size=0.5)

    # ------------------------------------------------------------------------------------#
    # Word Embedding
    def spacy_word_embedding(self, dataset):
        text = []
        vectors = []
        count = 0
        increments = 100000
        checkpoint = increments
        remaining = len(dataset['text'])
        starttime = time.time()
        
        for doc in self.nlp.pipe(dataset['text'].tolist(), n_process=8, disable=['tok2vec', 'parser', 'senter', 'ner']):
            vectors.append(doc.vector)
            text.append(self.clean_text(doc))
            count += 1
            if count == checkpoint:
                endtime = time.time()
                print("Remaining preprocess data:", remaining, " Time past:", endtime-starttime, end='\r')
                checkpoint += increments
                remaining -= increments
                starttime = time.time()
                self.yelp['text'] += text
                self.yelp['vectors'] += vectors
                
                self.save_embedded_json(self.yelp)
                text = []
                vectors = []
        return text, vectors
    
    def clean_text(self, doc):
        lowercase = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in doc]
        lemmatize = [word for word in lowercase if word not in stop_words.STOP_WORDS and word not in string.punctuation]
        text = " ".join(lemmatize)
        return text

    def count_vectorize(self, dataset):
        vec = CountVectorizer()
        return vec.fit_transform(dataset['text'])

    #############################################################################################
    # Saving code
#     def save_model(self, model):
#         filename = "./Save/" + self.c_method + "_" + self.task + ".sav"
#         pickle.dump(model, open(filename, 'wb'))

#     # for custom saves such as feature selector
#     def save_feature_model(self, model, path):
#         print("Feature Selection Complete")
#         filename = path
#         pickle.dump(model, open(filename, 'wb'))
#         print("Feature Selector saved")

#     def load_model(self, path):
#         filename = path
#         if os.path.isfile(filename):
#             return pickle.load(open(filename, 'rb'))
#         else:
#             print("(" + path + ")" + "No such file exists, starting new model")

    def load_json(self):
        if self.filepath == './yelp_dataset/yelp_academic_dataset_review.json':
            if os.path.isfile(self.word_embedded_dataset_path) and self.preprocessed:
                if self.nrows == 0:
                    self.yelp = pd.read_json(self.word_embedded_dataset_path, lines=True)
                else:
                    self.yelp = pd.read_json(self.word_embedded_dataset_path, lines=True, nrows=self.nrows)
            else:
                if self.nrows == 0:
                    self.yelp = pd.read_json(self.filepath, lines=True)
                else:
                    self.yelp = pd.read_json('./yelp_dataset/yelp_academic_dataset_review.json', lines=True,
                                             nrows=self.nrows)
        else:
            if os.path.isfile(self.filepath):
                if self.nrows == 0:
                    self.yelp = pd.read_json(self.filepath, lines=True)
                else:
                    self.yelp = pd.read_json(self.filepath, lines=True, nrows=self.nrows)
            else:
                print("No such file exist")

    def save_embedded_json(self, dataset):
        dataset.to_json(self.word_embedded_dataset_path)
    #############################################################################################
