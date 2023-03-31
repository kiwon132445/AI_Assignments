import pandas as pd
import numpy as np
import sklearn
import pickle
import os.path
import json
import torch

import gensim
from gensim.models import KeyedVectors

import spacy
from spacy.pipeline.lemmatizer import Lemmatizer
import en_core_web_lg

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#Initialization method
class Yelp_Manager:
    #------------------------------------------------------------------------------------#
    # Initialized Variables
    input_columns = ['text']
    irrelevant_columns = ['review_id', 'user_id', 'business_id', 'date']
    output_columns = ['stars', 'useful', 'funny', 'cool']
    
    # Uninitialized Variables
    #     file_name
    #     model_name
    
    #     yelp

    #     x, y
    #     X_train, y_train
    #     X_valid, y_valid
    #     X_test, y_test
    #------------------------------------------------------------------------------------#
    
    
    def __init__(self, file_name, model_name="", nrows=0):
        self.file_name = file_name
        self.model_name = model_name
        self.nrows = nrows
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nlp = spacy.load("en_core_web_md")
        self.transformer = pipeline("sentiment-analysis", model="roberta-base")
    
    def load_dataset(self):
        self.load_json()
        self.convert_yelp()
        
    def process_dataset(self):
        self.x_y_split();
        self.train_valid_test_split();
    
    def load_json(self):
        if self.nrows == 0:
            self.yelp = pd.read_json('./yelp_dataset/yelp_academic_dataset_review.json', lines=True)
        else:
            self.yelp = pd.read_json('./yelp_dataset/yelp_academic_dataset_review.json', lines=True, nrows=self.nrows)
        
    def convert_yelp(self):
        #columns
        #review_id user_id business_id stars useful funny cool text date
        self.yelp['text'] = self.yelp['text'].astype('str')
        self.yelp = self.yelp.drop(columns=self.irrelevant_columns)
    
    def x_y_split(self):
        self.x = self.yelp.drop(columns=self.output_columns)
        self.y = self.yelp.drop(columns=self.input_columns)
        
    def train_valid_test_split(self):
        self.X_train, X_remaining, self.y_train, y_remaining = train_test_split(self.x, self.y, train_size=0.8) 
        self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(X_remaining, y_remaining, test_size=0.5)
        
    def word_embedding(self, text):
        doc = self.nlp(text)
        return doc.vector
    
    #############################################################################################
    # Saving code
    def save_model(self, model):
        filename = "./Save/" + self.c_method + "_" + self.task + ".sav"
        pickle.dump(model, open(filename, 'wb'))
    
    #for custom saves such as feature selector
    def save_feature_model(self, model, path):
        print("Feature Selection Complete")
        filename = path
        pickle.dump(model, open(filename, 'wb'))
        print("Feature Selector saved")
    
    def load_model(self, path):
        filename = path
        if os.path.isfile(filename):
            return pickle.load(open(filename, 'rb'))
        else:
            print("(" + path + ")" + "No such file exists, starting new model")
    #############################################################################################