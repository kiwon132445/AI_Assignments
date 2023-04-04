import pandas as pd
import pickle
import os.path
import torch
import time
import string
from tqdm import tqdm
import json
import numpy as np
from tabulate import tabulate

import spacy
from spacy.lang.en import stop_words
from spacy.pipeline.lemmatizer import Lemmatizer
from spacy.language import Language
from spacy.tokens import DocBin

import sklearn
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from Bi_LSTM import *


class Yelp_Manager:
    # ------------------------------------------------------------------------------------#
    # Initialized Variables
    text = []
    vectors = []
    doc = []
    
    x = None
    yelp = None
    # columns
    # review_id user_id business_id stars useful funny cool text date
    input_columns = ['text']
    irrelevant_columns = []
    output_columns = ['stars', 'useful', 'funny', 'cool']
    
    clean_text_dataset_path_json = './Save/yelp_academic_dataset_review(clean_text).json'
    word_embedded_dataset_path_json = './Save/yelp_academic_dataset_review(word_embedded)'
    
    #word_embedded_dataset_path_spacy = './Save/yelp_academic_dataset_review(word_embedded).spacy'
    bi_lstm_model_path = './Save/BiLSTM_trained_model.pt'

    # Uninitialized Variables
    #     filepath
    #     model_name

    #     yelp

    #     x, y
    #     X_train, y_train
    #     X_valid, y_valid
    #     X_test, y_test
    # ------------------------------------------------------------------------------------#

    def __init__(self, filepath='./yelp_dataset/yelp_academic_dataset_review.json', model="", model_path="", nrows=0, preprocessed=True, embedded=True):
        self.filepath = filepath
        self.model = model
        self.model_path = model_path
        self.nrows = nrows
        self.preprocessed = preprocessed
        self.embedded = embedded

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nlp = spacy.load("en_core_web_lg")
        self.doc_bin = DocBin()
    
    # Loads the file and converts content into useable datatype
    def initialize(self, load_text=False, load_vectors=False):
        self.load_json(load_text, load_vectors)
    
    # Preprocess the data if it wasn't already, split the x and y
    def preprocess_dataset(self, clean=True):
        if clean:
            self.x['text'] = self.spacy_preprocessing_text(self.x['text'])
            if not self.preprocessed:
                self.save_cleaned_json(self.x['text'])
        
    # Split the data into training, validation, and test sets
    def process_dataset(self, vectorize=False):
        if vectorize:
            self.x['vectors'] = self.spacy_vectorize(self.x['text'])
            if not self.embedded:
                self.save_embedded_json(self.x['vectors'])
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
        self.x = self.yelp.drop(columns=self.output_columns)
        self.y = self.yelp.drop(columns=self.input_columns)
    
    # Split the data into training, validation, and test sets
    def train_valid_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, train_size=0.8)
#         self.X_train, X_remaining, self.y_train, y_remaining = train_test_split(self.x, self.y, train_size=0.8)
#         self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(X_remaining, y_remaining, test_size=0.5)

    # ------------------------------------------------------------------------------------#
    # Word Preprocessing
    def spacy_preprocessing_text(self, dataset_text):
        i_tqdm = tqdm(total=len(dataset_text))
        i_tqdm.set_description('Spacy cleaning')
        for doc in self.nlp.pipe(dataset_text.tolist(), n_process=10, disable=['tok2vec', 'parser', 'senter', 'ner']):
            clean_text = self.clean_text(doc)
            self.text.append(clean_text)
            i_tqdm.update(1)
        i_tqdm.close()
        return self.text
    
#     def spacy_preprocessing_doc(self, dataset_text):
#         i_tqdm = tqdm(total=len(dataset_text))
#         i_tqdm.set_description('Spacy cleaning and embedding')
#         for doc in self.nlp.pipe(dataset_text.tolist(), n_process=10, disable=['tok2vec', 'parser', 'senter', 'ner']):
#             clean_text = self.clean_text(doc)
#             self.doc_bin.add(self.nlp(clean_text))
#             i_tqdm.update(1)
#         i_tqdm.close()        
#         return self.doc_bin
    
    def clean_text(self, doc):
        lowercase = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in doc]
        lemmatize = [word for word in lowercase if word not in stop_words.STOP_WORDS and word not in string.punctuation]
        text = " ".join(lemmatize)
        return text
    
    # ------------------------------------------------------------------------------------#
    # Word Embedding
    
    # Count vectorizer
    def count_vectorize(self, text_dataset):
        vec = CountVectorizer(ngram_range=(1, 1))
        return vec.fit_transform(text_dataset)
    
    def spacy_vectorize(self, dataset_text):
        i_tqdm = tqdm(total=len(dataset_text))
        i_tqdm.set_description('Spacy Embedding text')
        for doc in self.nlp.pipe(dataset_text.tolist(), n_process=10, disable=['tok2vec', 'tagger', 'parser', 'senter', 'attribute_ruler', 'lemmatizer', 'ner']):
            self.vectors.append(doc.vector)
            i_tqdm.update(1)
        i_tqdm.close()   
        return self.vectors
        
    # ------------------------------------------------------------------------------------#
    #######################################################################################
    #######################################################################################
    # ------------------------------------------------------------------------------------#
    # Models
    def run_model(self):
        self.initialize()
        self.preprocess_dataset()
        
        if self.model == "RFC":
            self.run_randomForestClassifier()
        elif self.model == "BILSTM":
            self.run_Bi_LSTM()
        elif self.model == "NB":
            self.run_NB()
        elif self.model == "KNC":
            self.run_KNeighborsClassifier()
    
    def run_NB(self):
        self.process_dataset()
        
        x = self.x
        v = CountVectorizer(ngram_range=(1, 1))
        x = v.fit_transform(x['text'])

        unigram_tf_idf_transformer = TfidfTransformer()
        unigram_tf_idf_transformer.fit(x)

        x = unigram_tf_idf_transformer.transform(x)
        y = self.y

        nb_stars = NBClassifier()
        nb_useful = NBClassifier()
        nb_funny = NBClassifier()
        nb_cool = NBClassifier()

        nb_stars_score = nb_stars.get_trained_data(x, y["stars"])
        nb_useful_score = nb_useful.get_trained_data(x, y["useful"])
        nb_funny_score = nb_funny.get_trained_data(x, y["funny"])
        nb_cool_score = nb_cool.get_trained_data(x, y["cool"])

        table_MultinomialNB = ['Score (Stars)', 'Score (Useful)', 'Score (Funny)', 'Score (Cool)'], [nb_stars_score, nb_useful_score, nb_funny_score, nb_cool_score]

        print('Table For Naive Bayes Probabilistic Classifier')
        print(tabulate(table_MultinomialNB, headers='firstrow'))
    
    # Bi_LSTM (RNN) (Bidirectional Long-short term memory)
    def run_Bi_LSTM(self):
        self.process_dataset(vectorize=True)
        manager = Bi_LSTM_Manager(hidden_layers=512)
        manager.model = self.load_BiLSTM_model()
        
        if manager.model is None:
            manager.model = Bi_LSTM_Manager(hidden_layers=512)
        manager.train_model(self.X_train, self.y_train, loss_function=1, optimizer=0)
        
        y_preds = manager.predict(self.X_test)
        
        scores = manager.f1_score(self.y_test, y_preds, average='macro')
        
        print("f1-score(Macro)\n")
        for key in scores:
            print(key, ": %.2f" % (scores[key]*100), "%")
    
    def run_randomForestClassifier(self):
        x = self.x
        y = self.y

        y_stars = y.drop(columns=['useful', 'funny', 'cool'])
        y_useful = y.drop(columns=['stars', 'funny', 'cool'])
        y_funny = y.drop(columns=['stars', 'useful', 'cool'])
        y_cool = y.drop(columns=['stars', 'useful', 'funny'])
        
        v = CountVectorizer()
        x = v.fit_transform(x['text'])
        
        x_train_stars, x_remaining_stars, y_train_stars, y_remaining_stars = train_test_split(x, y_stars, train_size=0.8) 
        x_train_useful, x_remaining_useful, y_train_useful, y_remaining_useful = train_test_split(x, y_useful, train_size=0.8) 
        x_train_funny, x_remaining_funny, y_train_funny, y_remaining_funny = train_test_split(x, y_funny, train_size=0.8) 
        x_train_cool, x_remaining_cool, y_train_cool, y_remaining_cool = train_test_split(x, y_cool, train_size=0.8)
        
        rf = RandomForestClassifier(max_depth=10)
        rf2 = RandomForestClassifier(max_depth=10)
        rf3 = RandomForestClassifier(max_depth=10)
        rf4 = RandomForestClassifier(max_depth=10)
        
        y_train_stars = np.ravel(y_train_stars)
        y_train_useful = np.ravel(y_train_useful)
        y_train_funny = np.ravel(y_train_funny)
        y_train_cool = np.ravel(y_train_cool)

        rf_fit_stars = rf.fit(x_train_stars, y_train_stars)
        rf_fit_useful = rf2.fit(x_train_useful, y_train_useful)
        rf_fit_funny = rf3.fit(x_train_funny, y_train_funny)
        rf_fit_cool = rf4.fit(x_train_cool, y_train_cool)
        
        y_pred_class_RF_stars = rf.predict(x_remaining_stars)
        y_pred_class_RF_useful = rf2.predict(x_remaining_useful)
        y_pred_class_RF_funny = rf3.predict(x_remaining_funny)
        y_pred_class_RF_cool = rf4.predict(x_remaining_cool)
        
        score_stars = metrics.accuracy_score(y_remaining_stars, y_pred_class_RF_stars)
        score_useful = metrics.accuracy_score(y_remaining_useful, y_pred_class_RF_useful)
        score_funny = metrics.accuracy_score(y_remaining_funny, y_pred_class_RF_funny)
        score_cool = metrics.accuracy_score(y_remaining_cool, y_pred_class_RF_cool)
        
        table_RFC = ['Score (Stars)', 'Score (Useful)', 'Score (Funny)', 'Score (Cool)'],\
        [score_stars, score_useful, score_funny, score_cool]

        print('Table For RandomForestClassifier \n')
        print(tabulate(table_RFC, headers='firstrow'))
        
    def run_KNeighborsClassifier(self):
        KNN = KNeighborsClassifier()
        KNN2 = KNeighborsClassifier()
        KNN3 = KNeighborsClassifier()
        KNN4 = KNeighborsClassifier()

        KNN_fit_stars = KNN.fit(self.X_train, self.y_train['stars'])
        KNN_fit_useful = KNN2.fit(self.X_train, self.y_train['useful'])
        KNN_fit_funny = KNN3.fit(self.X_train, self.y_train['funny'])
        KNN_fit_cool = KNN4.fit(self.X_train, self.y_train['cool'])
        
        y_pred_class_KNN_stars = KNN.predict(self.X_test)
        y_pred_class_KNN_useful = KNN2.predict(self.X_test)
        y_pred_class_KNN_funny = KNN3.predict(self.X_test)
        y_pred_class_KNN_cool = KNN4.predict(self.X_test)
        
        score_stars_KNN = metrics.accuracy_score(self.y_test['stars'], y_pred_class_KNN_stars)
        score_useful_KNN = metrics.accuracy_score(self.y_test['useful'], y_pred_class_KNN_useful)
        score_funny_KNN = metrics.accuracy_score(self.y_test['funny'], y_pred_class_KNN_funny)
        score_cool_KNN = metrics.accuracy_score(self.y_test['cool'], y_pred_class_KNN_cool)

        table_KNN = ['Score (Stars)', 'Score (Useful)', 'Score (Funny)', 'Score (Cool)'], [score_stars_KNN, score_useful_KNN, score_funny_KNN, score_cool_KNN]

        print('Table For K-Nearest Neighbour \n')
        print(tabulate(table_KNN, headers='firstrow'))
    # ------------------------------------------------------------------------------------#
    # Techniques
    

    #############################################################################################
    # Saving and loading data
    
    # Save model
    def save_model(self, model):
        filename = "./Save/" + self.model + "_" + self.technique + ".sav"
        pickle.dump(model, open(filename, 'wb'))
        
    def save_BiLSTM_model(self, model):
        torch.save(model, self.bi_lstm_model_path)
    
    def load_BiLSTM_model(self):
        if not self.model_path == "":
            self.bi_lstm_model_path = self.model_path
        if not os.path.isfile(self.bi_lstm_model_path):
            print("Torch model in given path does not exist")
            return -1
        model = torch.load(self.bi_lstm_model_path)
        model.eval()
        return model
    
    # Load model
    def load_model(self, path):
        filename = path
        if os.path.isfile(filename):
            return pickle.load(open(filename, 'rb'))
        else:
            print("(" + path + ")" + "No such file exists, starting new model")
    
    # Save dataset from json
    def load_json(self, load_text, load_vectors):
        if self.filepath == './yelp_dataset/yelp_academic_dataset_review.json':
            #load data
            if self.nrows == 0:
                self.yelp = pd.read_json(self.filepath, lines=True)
            else:
                self.yelp = pd.read_json(self.filepath, lines=True, nrows=self.nrows)
            
            self.convert_yelp()
            self.x_y_split()
            
            #load clean text
            if os.path.isfile(self.clean_text_dataset_path_json) and self.preprocessed and load_text:
                if self.nrows == 0:
                    self.x['text'] = pd.read_json(self.clean_text_dataset_path_json, lines=True)
                else:
                    self.x['text'] = pd.read_json(self.clean_text_dataset_path_json, lines=True, nrows=self.nrows)
                    
            #load embedded data
            if self.preprocessed and load_vectors:
                self.load_embedded_json(self.nrows)
        else:
            if os.path.isfile(self.filepath):
                if self.nrows == 0:
                    self.yelp = pd.read_json(self.filepath, lines=True)
                else:
                    self.yelp = pd.read_json(self.filepath, lines=True, nrows=self.nrows)
                
                self.convert_yelp()
                self.x_y_split()
            else:
                print("No such file in the provided path")
                return -1
    
    # Save word embedded dataset
    def save_embedded_json(self, vector_dataset):
        i_tqdm = tqdm(range(1, len(vector_dataset) + 1))
        every = 1000000
        start = 0
        end = start+every
        i = 1
        i_tqdm.set_description('Saving Spacy Embedding: ')
        for j in range(10):
            pathname = self.word_embedded_dataset_path_json + '[' + str(i) + '].json'
            if end > len(vector_dataset)+1:
                end = len(vector_dataset)+1
            
            if not os.path.isfile(pathname) or not self.embedded:
                pd.DataFrame(vector_dataset[start:end]).to_json(pathname, orient="records", lines=True)
                
            if end == len(vector_dataset):
                i_tqdm.update(end)
                break
            i+=1
            start += every
            end += every
            
            i_tqdm.update(end)
        i_tqdm.close()
    
    def load_embedded_json(self, nrows):
        every = 1000000
        remaining_lines = nrows
        getLines = every
        i = 1
        df = pd.DataFrame()
        self.x = self.x.drop(columns=['text'])
        
        i_tqdm = tqdm(range(10))
        i_tqdm.set_description('Saving Spacy Embedding: ')
        for j in range(10):
            pathname = self.word_embedded_dataset_path_json + '[' + str(i) + '].json'
            if not os.path.isfile(pathname):
                break
            
            if nrows == 0:
                if 'vectors' not in df.columns:
                    df['vectors'] = pd.read_json(pathname, lines=True)
                else:
                     df = pd.concat([df['vectors'], pd.read_json(pathname, lines=True)], axis=1)
            else:
                if remaining_lines < 1000000:
                    getLines = remaining_lines
                
                if 'vectors' not in df.columns:
                    df['vectors'] = pd.read_json(pathname, lines=True, nrows=getLines)
                else:
                    df = pd.concat([df['vectors'], pd.read_json(pathname, lines=True, nrows=getLines)], axis=1)
                
                remaining_lines-=every
                if remaining_lines <= 0:
                    break
            i_tqdm.update(1)
            i+=1
        i_tqdm.close()
        self.x = pd.concat([self.x, df], axis=1)

    # Save word embedded dataset
    def save_cleaned_json(self, dataset):
        if not os.path.isfile(self.clean_text_dataset_path_json):
            dataset.to_json(self.clean_text_dataset_path_json, orient="records", lines=True)
    #############################################################################################
    
class NBClassifier:
    def __init__(self):
        pass

    def get_trained_data(self, x, y):
        X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8) 
        # X_valid, X_test, y_valid, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5)

        # Train the Naive Bayes classifier
        nb_classifier = MultinomialNB(alpha = 0.1)
        nb_classifier.fit(X_train, y_train)

        # Predict the sentiment for the test data
        y_pred = nb_classifier.predict(X_test)

        # Evaluate the performance of the classifier
        print(metrics.accuracy_score(y_test, y_pred))
        return metrics.accuracy_score(y_test, y_pred)
