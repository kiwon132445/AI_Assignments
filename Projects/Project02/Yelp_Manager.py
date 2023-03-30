import pandas as pd
import sklearn
import pickle
import os.path
import json
import torch

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#Initialization method
class Yelp_Manager:
    #------------------------------------------------------------------------------------#
    # Initialized Variables
    input_columns = ['review_id', 'user_id', 'business_id', 'text', 'date']
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
    
    
    def __init__(self, file_name, model_name=""):
        self.file_name = file_name
        self.model_name = model_name
    
    def program_runner(self):
        self.load_json()
        self.convert_yelp()
        self.x_y_split();
        self.train_valid_test_split();
    
    def load_json(self):
        chunks = pd.read_json('./yelp_dataset/yelp_academic_dataset_review.json', lines=True, chunksize = 10000)
        self.yelp = pd.DataFrame()
        for chunk in chunks:
          self.yelp = pd.concat([self.yelp, chunk])
        
    def convert_yelp(self):
        #columns
        #review_id user_id business_id stars useful funny cool text date
        
#         self.yelp['review_id'] = self.yelp['review_id'].astype('str')
#         self.yelp['user_id'] = self.yelp['user_id'].astype('str')
#         self.yelp['business_id'] = self.yelp['business_id'].astype('str')
        self.yelp['text'] = self.yelp['text'].astype('str')
        self.yelp.dropna
    
    def x_y_split(self):
        self.x = self.yelp.drop(columns=self.output_columns)
        self.y = self.yelp.drop(columns=self.input_columns)
        self.y = self.y.drop(columns=self.irrelevant_columns)
        
    def train_valid_test_split(self):
        self.X_train, X_remaining, self.y_train, y_remaining = train_test_split(self.x, self.y, train_size=0.8) 
        self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(X_remaining, y_remaining, test_size=0.5) 
    
    
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