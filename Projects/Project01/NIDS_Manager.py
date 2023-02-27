import pandas as pd
import sklearn
import pickle

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from feature_analysis import *
from classifier import *

class NIDS_Manager:
    goal_columns = ['attack_cat', 'Label']
    
    #Initialization method
    def __init__(self, csv_name, c_method, task, model_name=""):
        self.csv_name = csv_name
        self.c_method = c_method
        self.task = task
        self.model_name = model_name
        
        self.setup()
        self.feature_analysis_init("RFECV")
        self.classifier_init()
    
    #split csv into x and y with 'goal_columns' as y
    def x_and_y_split(self):
        self.x = self.nids.drop(columns=self.goal_columns)
        self.y_attack_cat = self.nids['attack_cat']
        self.y_label = self.nids['Label']
    
    #get set x value
    def get_x(self):
        return self.x
    def set_x(self, x):
        self.x = x
    
    #get set y value
    def get_y(self):
        return self.y_attack_cat, self.y_label
    def set_y_attack_cat(self, y):
        self.y_attack_cat = y
    def set_y_label(self, y):
        self.y_label = y
    
    def train_test_split(self):
        self.x_train, self.x_test, self.y_train_attack, self.y_test_attack, self.y_train_label, self.y_test_label = train_test_split(self.x, self.y_attack_cat, self.y_label, test_size=0.3, random_state=1)
    
    #return the csv content
    def get_csv(self):
        return self.nids
    
    #general setup
    def setup(self):
        self.load_csv()
        self.convert_csv()
        self.x_and_y_split()
        self.train_test_split()
        
    #load csv
    def load_csv(self):
        self.nids = pd.read_csv(self.csv_name, converters={'attack_cat': str.strip}, low_memory=False)
    
    #convert csv content into analysis and machine learning friendly specifics
    def convert_csv(self):
        #converting null values to str
        self.nids['ct_flw_http_mthd'] = self.nids['ct_flw_http_mthd'].astype('str')
        self.nids['ct_flw_http_mthd'] = self.nids['ct_flw_http_mthd'].astype('str')
        self.nids['is_ftp_login'] = self.nids['is_ftp_login'].astype('str')
        self.nids['ct_ftp_cmd'] = self.nids['ct_ftp_cmd'].astype('str')

        self.nids["sport"] = pd.to_numeric(self.nids["sport"], errors="coerce")
        self.nids["dsport"] = pd.to_numeric(self.nids["dsport"], errors="coerce")

        #converting str to int
        #self.nids['attack_cat'] = pd.factorize(self.nids['attack_cat'])[0]
        self.nids['proto'] = pd.factorize(self.nids['proto'])[0]
        self.nids['state'] = pd.factorize(self.nids['state'])[0]
        self.nids['service'] = pd.factorize(self.nids['service'])[0]

        self.nids['ct_flw_http_mthd'] = pd.factorize(self.nids['ct_flw_http_mthd'])[0]
        self.nids['ct_flw_http_mthd'] = pd.factorize(self.nids['ct_flw_http_mthd'])[0]
        self.nids['is_ftp_login'] = pd.factorize(self.nids['is_ftp_login'])[0]
        self.nids['ct_ftp_cmd'] = pd.factorize(self.nids['ct_ftp_cmd'])[0]

        self.nids['srcip'] = preprocessing.LabelEncoder().fit_transform(self.nids['srcip'])
        self.nids['dstip'] = preprocessing.LabelEncoder().fit_transform(self.nids['dstip'])
        
        self.nids.dropna(axis='rows', subset=['attack_cat', 'sport', 'dsport'], inplace=True)
    
    ##########################################################################################
    #feature analysis initializer
    def feature_analysis_init(self, fa_type):
        self.fa = FeatureAnalysis()
        if fa_type == 'RFECV':
            self.fa.rfecv_init()
    #
    #
    #feature analysis (Recursive Feature Elimination with Logistic Regression)
    def rfecv_fit(self):
        self.fa.rfecv_init()
        
        self.fa.rfecv_fit(self.x_train, self.y_train_attack)
        self.fa.rfecv_fit(self.x_train, self.y_train_label)
    
    def rfecv_x(self):
        support = self.fa.rfecv_support_()
        return self.x[self.x.columns[support]]
    
    def rfecv_select_plot(self):
        self.fa.rfecv_select_plot()
    
    ##########################################################################################
    #Classifier initializer
    def classifier_init(self, c_type=""):
        self.classifier = Classifier()
        if c_type == "":
            if self.c_method == 'LRCV':
                self.classifier.logistic_init()
            elif self.c_method == 'SVC':
                self.classifier.svc_init()
        else:
            if c_type == 'LRCV':
                self.classifier.logistic_init()
            elif c_type == 'SVC':
                self.classifier.svc_init()
    
    #
    #
    #Classifier (Logistic Regression CV)
    def run_logistic(self, x_train, x_test, y_train, y_test):
        self.logistic_fit(x_train, y_train)
        pred = self.logistic_pred(x_test)
        score = self.logistic_score(x_test, y_test)
        return pred, score
    
    def logistic_fit(self, x_train, y_train):
        self.classifier.logistic_fit(x_train, y_train)
    def logistic_pred(self, x_test):
        return self.classifier.logistic_pred(x_test)
    def logistic_score(self, x, y):
        return self.classifier.logistic_score(x, y)
    
    #
    #
    #Classifier Support Vector Classification
    def run_svc(self, x_train, x_test, y_train, y_test):
        self.svc_fit(x_train, y_train)
        pred = self.svc_pred(x_test)
        score = self.svc_score(x_test, y_test)
        return pred, score
    
    def svc_fit(self, x_train, y_train):
        self.classifier.svc_fit(x_train, y_train)
    def svc_pred(self, x_test):
        return self.classifier.svc_pred(x_test)
    def svc_score(self, x_test, y_test):
        return self.classifier.svc_score(x_test, y_test)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        