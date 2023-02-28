from feature_analysis import *
from classifier import *

import pandas as pd
import sklearn
import pickle
<<<<<<< HEAD
import os.path
=======
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

<<<<<<< HEAD
# from sklearn.linear_model import LinearRegression

# from sklearn.datasets import make_classification
# from sklearn.neural_network import MLPClassifier

=======
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
class NIDS_Manager:
    goal_columns = ['attack_cat', 'Label']
    
    #Initialization method
<<<<<<< HEAD
    def __init__(self, csv_name, c_method, task, model_name="", feature_name="LCV"):
=======
    def __init__(self, csv_name, c_method, task, model_name="", feature_name="RFECV"):
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
        self.csv_name = csv_name
        self.c_method = c_method
        self.task = task
        self.model_name = model_name
        self.feature_name = feature_name
        
<<<<<<< HEAD
        if self.classifier_init() == -1:
            return -1
        if self.feature_analysis_init(feature_name):
            return -1
        
        self.setup()
        self.feature_selector()
        self.command_line_runner()
=======
        self.command_line_runner()
        self.setup()
        if self.classifier_init() == -1:
            return -1
        self.feature_analysis_init(feature_name)
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
    
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
<<<<<<< HEAD
        if self.feature_name == "LCV":
            if self.task == "attack_cat":
                self.y_unique_names = self.nids['attack_cat'].unique()
            else:
                self.y_unique_names = self.nids['Label'].unique().astype('str')
            self.nids['attack_cat'] = pd.factorize(self.nids['attack_cat'])[0]
        
=======
        #self.nids['attack_cat'] = pd.factorize(self.nids['attack_cat'])[0]
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
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
    
    #----------------------------------------------#
    #split csv into x and y with 'goal_columns' as y
    def x_and_y_split(self):
        self.x = self.nids.drop(columns=self.goal_columns)
        self.y_attack_cat = self.nids['attack_cat']
        self.y_label = self.nids['Label']
    
    #split data into training and testing sets
    def train_test_split(self):
<<<<<<< HEAD
        self.x_train, self.x_test, self.y_train_attack, self.y_test_attack, self.y_train_label, self.y_test_label = train_test_split(self.x, self.y_attack_cat, self.y_label, test_size=0.3)
    
    #----------------------------------------------#
    def save_model(self, model):
        filename = "./Save/" + self.c_method + "_" + self.task + ".sav"
        pickle.dump(model, open(filename, 'wb'))
    
    def load_model(self):
        filename = "./Save/" + self.c_method + "_" + self.task + ".sav"
        if os.path.isfile(filename):
            return pickle.load(filename)
        else:
            print("No such file exists, starting new model")
            return -1
=======
        self.x_train, self.x_test, self.y_train_attack, self.y_test_attack, self.y_train_label, self.y_test_label = train_test_split(self.x, self.y_attack_cat, self.y_label, test_size=0.3, random_state=1)
    
    #----------------------------------------------#
    def save_model(self, model):
        filename = "./Save/" + self.c_method + "_" + task
        pickle.dump(model, open(filename, 'wb'))
    
    def load_model(self):
        filename = "./Save/" + self.c_method + "_" + task
        return pickle.load(filename)
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
    
    ##########################################################################################
    #
    # Command line runner
    def command_line_runner(self):
<<<<<<< HEAD
        task = self.task_checker()
        
#         if self.feature_name != "":
#             if task == 1:
#                 if self.feature_name == 'RFECV':
#                     self.run_rfecv(self.x, self.y_attack_cat)
#                 elif self.feature_name == 'LCV':
#                     self.run_LassoCV(self.x, self.y_attack_cat)
#             elif task == 2:
#                 if self.feature_name == 'RFECV':
#                     self.run_rfecv(self.x, self.y_label)
#                 elif self.feature_name == 'LCV':
#                     self.run_LassoCV(self.x, self.y_label)
#             else:
#                 return -1
    
        if self.c_method == 'LRCV':
            self.LRCV_runner(task)
        elif self.c_method == 'SVC':
            self.SVC_runner(task)
        elif self.c_method == 'GNB':
            self.GNB_runner(task)
        elif self.c_method == 'DTC':
            self.DTC_runner(task)
        elif self.c_method == 'ADA':
            self.ADA_runner(task)
        elif self.c_method == 'KN':
            self.KN_runner(task)
=======
        if self.c_method == 'LRCV':
            self.LRCV_runner()
        elif self.c_method == 'SVC':
            self.SVC_runner()
        elif self.c_method == 'GNB':
            self.GNB_runner()
        elif self.c_method == 'DTC':
            self.DTC_runner()
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
    
    def task_checker(self):
        if self.task == 'attack_cat':
            return 1
        elif self.task == 'label':
            return 2
        else:
            print('No such task exists in code\n\
            options:\n\
            \t - attack_cat\n\
            \t - label')
            
            return -1
<<<<<<< HEAD
    
    def feature_selector(self):
        task = self.task_checker()
        if self.feature_name != "":
            if task == 1:
                if self.feature_name == 'RFECV':
                    self.run_rfecv(self.x, self.y_attack_cat)
                elif self.feature_name == 'LCV':
                    self.run_LassoCV(self.x, self.y_attack_cat)
            elif task == 2:
                if self.feature_name == 'RFECV':
                    self.run_rfecv(self.x, self.y_label)
                elif self.feature_name == 'LCV':
                    self.run_LassoCV(self.x, self.y_label)
            else:
                return -1
    #----------------------------------------------#
    def LRCV_runner(self, task):
        print("Logistic Regression CV Running...")
        
        if self.model_name != "":
            model = self.load_model(model_name)
            if model == -1:
                self.classifier.log_reg = model
            
        # attack_cat
        if task == 1:
            pred, score = self.run_logistic(self.x_train, self.x_test, self.y_train_attack, self.y_test_attack)
            self.print_class_report(pred, self.y_test_attack)
        # label
        else:
            pred, score = self.run_logistic(self.x_train, self.x_test, self.y_train_label, self.y_test_label)
            self.print_class_report(pred, self.y_test_label)
    
    def SVC_runner(self, task):
        print("Linear Support Vector Classification Running...")
        
        if self.model_name != "":
            model = self.load_model(model_name)
            if model == -1:
                self.classifier.svc = model
        
        # attack_cat
        if task == 1:
            pred, score = self.run_svc(self.x_train, self.x_test, self.y_train_attack, self.y_test_attack)
            self.print_class_report(pred, self.y_test_attack)
        # label
        else:
            pred, score = self.run_svc(self.x_train, self.x_test, self.y_train_label, self.y_test_label)
            self.print_class_report(pred, self.y_test_label)
    
    def GNB_runner(self, task):
        print("Gaussian Naive Bayes Running...")
        
        if self.model_name != "":
            model = self.load_model(model_name)
            if model == -1:
                self.classifier.log_reg = model
        
        # attack_cat
        if task == 1:
            pred = self.run_gauss(self.x_train, self.x_test, self.y_train_attack, self.y_test_attack)
            self.print_class_report(pred, self.y_test_attack)
        # label
        else:
            pred = self.run_gauss(self.x_train, self.x_test, self.y_train_label, self.y_test_label)
            self.print_class_report(pred, self.y_test_label)
    
    def DTC_runner(self, task):
        print("Decision Tree Classifier Running...")
        
        if self.model_name != "":
            model = self.load_model(model_name)
            if model == -1:
                self.classifier.dtc = model
        
        # attack_cat
        if task == 1:
            pred = self.run_dtc(self.x_train, self.x_test, self.y_train_attack, self.y_test_attack)
            self.print_class_report(pred, self.y_test_attack)
        # label
        else:
            pred = self.run_dtc(self.x_train, self.x_test, self.y_train_label, self.y_test_label)
            self.print_class_report(pred, self.y_test_label)
            
    def ADA_runner(self, task):
        print("AdaBoost Classification Running...")
        
        if self.model_name != "":
            model = self.load_model(model_name)
            if model == -1:
                self.classifier.ada = model
        
        # attack_cat
        if task == 1:
            pred = self.run_ada(self.x_train, self.x_test, self.y_train_attack, self.y_test_attack)
            self.print_class_report(pred, self.y_test_attack)
        # label
        else:
            pred = self.run_ada(self.x_train, self.x_test, self.y_train_label, self.y_test_label)
            self.print_class_report(pred, self.y_test_label)
    
    def KN_runner(self, task):
        print("K Neighbors Classifier Running...")
        
        if self.model_name != "":
            model = self.load_model(model_name)
            if model == -1:
                self.classifier.ada = model
        
        # attack_cat
        if task == 1:
            pred = self.run_kn(self.x_train, self.x_test, self.y_train_attack, self.y_test_attack)
            self.print_class_report(pred, self.y_test_attack)
        # label
        else:
            pred = self.run_kn(self.x_train, self.x_test, self.y_train_label, self.y_test_label)
            self.print_class_report(pred, self.y_test_label)
=======
    #----------------------------------------------#
    def LRCV_runner(self):
        task = self.task_checker()
        # attack_cat
        if task == 1:
            return 1
        # label
        elif task == 2:
            return 2
        return -1
    
    def SVC_runner(self):
        task = self.task_checker()
        # attack_cat
        if task == 1:
            return 1
        # label
        elif task == 2:
            return 2
        return -1
    
    def GNB_runner(self):
        task = self.task_checker()
        # attack_cat
        if task == 1:
            return 1
        # label
        elif task == 2:
            return 2
        return -1
    
    def DTC_runner(self):
        task = self.task_checker()
        # attack_cat
        if task == 1:
            return 1
        # label
        elif task == 2:
            return 2
        return -1
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
    
    ##########################################################################################
    #feature analysis initializer
    def feature_analysis_init(self, fa_type):
        self.fa = FeatureAnalysis()
        if fa_type == 'RFECV':
            self.fa.rfecv_init()
<<<<<<< HEAD
        elif fa_type == 'PCA':
            self.fa.pca_init()
        elif fa_type == 'LCV':
            self.fa.LassoCV_init()
        elif fa_type == '':
            return 0
        else:
            return -1
        return 0
=======
        if fa_type == 'PCA':
            self.fa.pca_init()
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
    
    #----------------------------------------------#
    #
    #
<<<<<<< HEAD
    # Recursive Feature Elimination CV (RFECV) with Decision Tree Classification
    def run_rfecv(self, x, y):
        self.rfecv_fit(x, y)
        self.x_train = self.rfecv_x(self.x_train)
        self.x_test = self.rfecv_x(self.x_test)
    
=======
    #Recursive Feature Elimination CV (RFECV) with Logistic Regression
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
    def rfecv_fit(self, x, y):
        self.fa.rfecv_fit(x, y)
    
    def rfecv_x(self, x):
        return x[self.rfecv_selected_features()]
    
    def rfecv_x_specific_select(self, x, sf):
        return x[sf]
    
    def rfecv_selected_features(self):
        return self.x.columns[self.fa.rfecv_support_()]
    
    def rfecv_score(self, x, y):
        return self.fa.rfecv_score(x, y)
    
    def rfecv_plot(self):
        self.fa.rfecv_select_plot()
    
    #----------------------------------------------#
    #
    #
    #Principal Component Analysis
<<<<<<< HEAD
    def run_pca(self, x):
        ft = self.pca_fit_transform(x)
        return self.pca_x(ft)
        
=======
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
    def pca_fit_transform(self, x):
        return self.fa.pca_fit_transform(x)
    
    #remove the chose features from x
    def pca_x(self, x):
        return pd.DataFrame(data=x, columns=self.fa.pca.get_feature_names_out())
    
    #get explained variance
    def pca_get_e_variance(self):
        return self.fa.pca[1].explained_variance_
    
    #show explained variance bar
    def pca_e_var_bar(self):
        plt.bar(range(1,len(self.pca_get_e_variance())+1), self.pca_get_e_variance())

        plt.xlabel('PCA Feature')
        plt.ylabel('Explained variance')
        plt.title('Feature Explained Variance')
        plt.show()
<<<<<<< HEAD
    #----------------------------------------------#
    #
    # Linear Regression (LassoCV)
    def run_LassoCV(self, x, y):
        self.LassoCV_fit(x, y)
        important = self.LassoCV_important(x, y)
        self.x_train = self.x_train[important]
        self.x_test = self.x_test[important]
        return 0
    
    def LassoCV_fit(self, x, y):
        self.fa.LassoCV_fit(x, y);
    
    def LassoCV_important(self, x, y):
        return self.fa.LassoCV_important(x, y)
=======
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
    
    ##########################################################################################
    #Classifier initializer
    def classifier_init(self, c_type=""):
        self.classifier = Classifier()
        if c_type == "":
            if self.c_method == 'LRCV':
                self.classifier.logistic_init()
            elif self.c_method == 'SVC':
                self.classifier.svc_init()
            elif self.c_method == 'GNB':
                self.classifier.gauss_init()
            elif self.c_method == 'DTC':
                self.classifier.dtc_init()
<<<<<<< HEAD
            elif self.c_method == 'ADA':
                self.classifier.ada_init()
            elif self.c_method == 'KN':
                self.classifier.kn_init()
=======
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
            else:
                print('No such Classification method exists in code\n\
                ---------------------\n\
                options:\n\
                    \t - LRCV = Logistic Regression CV\n\
                    \t - SVC  = Support Vector Classification\n\
                    \t - GNB  = Gaussian Naive Bayes\n\
                    \t - DTC  = Decision Tree Classifier\n\
<<<<<<< HEAD
                    \t - ADA  = Ada Boost Classifier\n\
                    \t - KN   = KNeighborsClassifier\n\
=======
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
                ---------------------')
                return -1
        else:
            if c_type == 'LRCV':
                self.classifier.logistic_init()
            elif c_type == 'SVC':
                self.classifier.svc_init()
            elif c_type == 'GNB':
                self.classifier.gauss_init()
            elif c_type == 'DTC':
                self.classifier.dtc_init()
<<<<<<< HEAD
            elif c_type == 'ADA':
                self.classifier.ada_init()
            elif c_type == 'KN':
                self.classifier.kn_init()
=======
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
            else:
                print('No such Classification method exists in code\n\
                ---------------------\n\
                options:\n\
                    \t - LRCV = Logistic Regression CV\n\
                    \t - SVC  = Support Vector Classification\n\
                    \t - GNB  = Gaussian Naive Bayes\n\
                    \t - DTC  = Decision Tree Classifier\n\
<<<<<<< HEAD
                    \t - ADA  = Ada Boost Classifier\n\
                    \t - KN   = KNeighborsClassifier\n\
=======
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
                ---------------------')
                return -1
        return 0
    
    def print_model_acc_score(self, y_test, pred):
        print('Model accuracy score:  {0:0.4f}'. format(accuracy_score(y_test, pred)))
        
    def print_class_report(self, y_test, pred):
<<<<<<< HEAD
        print('Classification Report\n')
        if self.feature_name == "LCV":
            print(classification_report(y_test, pred, target_names=self.y_unique_names))
        else:
            print(classification_report(y_test, pred))
=======
        if c_type == 'LRCV':
            print('Classifier: Logistic Regression Cross-Validation (LRCV) \n')
        elif c_type == 'SVC':
            print('Classifier: Support Vector Classification (SVC) \n')
        elif c_type == 'GNB':
            print('Gaussian Naive Bayes (GNB) \n')
        elif c_type == 'DTC':
            print('Decision Tree Classifier (DTC) \n')
        
        print(classification_report(y_test, pred))
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
    
    #----------------------------------------------#
    #
    #
    # Logistic Regression CV (LRCV)
    def run_logistic(self, x_train, x_test, y_train, y_test):
        self.logistic_fit(x_train, y_train)
        pred = self.logistic_pred(x_test)
        score = self.logistic_score(x_test, y_test)
<<<<<<< HEAD
        
        self.save_model(self.classifier.log_reg)
=======
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
        return pred, score
    
    def logistic_fit(self, x_train, y_train):
        self.classifier.logistic_fit(x_train, y_train)
    def logistic_pred(self, x_test):
        return self.classifier.logistic_pred(x_test)
    def logistic_score(self, x, y):
        return self.classifier.logistic_score(x, y)
    
    #
    #
    # Support Vector Classification (SVC)
    def run_svc(self, x_train, x_test, y_train, y_test):
        self.svc_fit(x_train, y_train)
        pred = self.svc_pred(x_test)
        score = self.svc_score(x_test, y_test)
<<<<<<< HEAD
        
        self.save_model(self.classifier.svc)
=======
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
        return pred, score
    
    def svc_fit(self, x_train, y_train):
        self.classifier.svc_fit(x_train, y_train)
    def svc_pred(self, x_test):
        return self.classifier.svc_pred(x_test)
    def svc_score(self, x_test, y_test):
        return self.classifier.svc_score(x_test, y_test)
    
    #----------------------------------------------#
    #
    #
    # Gaussian Naive Bayes (GaussianNB)
<<<<<<< HEAD
    def run_gauss(self, x_train, x_test, y_train, y_test):
        self.gauss_fit(x_train, y_train)
        pred = self.gauss_pred(x_test)
        
        self.save_model(self.classifier.gauss)
        return pred
    
=======
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
    def gauss_fit(self, x, y):
        self.classifier.gauss_fit(x, y)
    
    def gauss_pred(self, x):
<<<<<<< HEAD
        return self.classifier.gauss_pred(x)
    #
    #
    # Decision Tree Classifier
    def run_dtc(self, x_train, x_test, y_train, y_test):
        self.dtc_fit(x_train, y_train)
        pred = self.dtc_pred(x_test)
        
        self.save_model(self.classifier.dtc)
        return pred
    
=======
        self.classifier.gauss_pred(x)
    #
    #
    # Decision Tree Classifier
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
    def dtc_fit(self, x, y):
        self.classifier.dtc_fit(x, y)
    
    def dtc_pred(self, x):
<<<<<<< HEAD
        return self.classifier.dtc_pred(x)
    
    #----------------------------------------------#
    #
    #
    # AdaBoost classification
    def run_ada(self, x_train, x_test, y_train, y_test):
        self.ada_fit(x_train, y_train)
        pred = self.ada_pred(x_test)
        
        self.save_model(self.classifier.ada)
        return pred
    
    def ada_fit(self, x, y):
        self.classifier.ada_fit(x, y)
        
    def ada_pred(self, x):
        return self.classifier.ada_pred(x)
    #
    #
    # K Neighbors Classifier
    def run_kn(self, x_train, x_test, y_train, y_test):
        self.kn_fit(x_train, y_train)
        pred = self.kn_pred(x_test)
        
        self.save_model(self.classifier.kn)
        return pred
    
    def kn_fit(self, x, y):
        self.classifier.kn_fit(x, y)
        
    def kn_pred(self, x):
        return self.classifier.kn_pred(x)
=======
        return self.classifier.dtc_pred(x)
>>>>>>> 751976cc28ba63cf5038f94dff151dbcb5709713
