import sys
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import sklearn
from sklearn import preprocessing

from feature_analysis import *

class NIDS_Class:
    
    #Initialization method
    def __init__(self, csv_name, c_method_name, task):
        self.csv_name = csv_name
        self.c_method_name = c_method_name
        self.task = task
        
        self.setup()
    
    #used when model name is included
    def add_model_name(model_name):
        self.model_name = model_name
    
    #return the csv content
    def get_csv(self):
        return self.nids
    
    #general setup
    def setup(self):
        self.load_csv()
        self.convert_csv()
        
    #load csv
    def load_csv(self):
        self.nids = pd.read_csv(self.csv_name)
    
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
        self.nids['attack_cat'] = pd.factorize(self.nids['attack_cat'])[0]
        self.nids['proto'] = pd.factorize(self.nids['proto'])[0]
        self.nids['state'] = pd.factorize(self.nids['state'])[0]
        self.nids['service'] = pd.factorize(self.nids['service'])[0]

        self.nids['ct_flw_http_mthd'] = pd.factorize(self.nids['ct_flw_http_mthd'])[0]
        self.nids['ct_flw_http_mthd'] = pd.factorize(self.nids['ct_flw_http_mthd'])[0]
        self.nids['is_ftp_login'] = pd.factorize(self.nids['is_ftp_login'])[0]
        self.nids['ct_ftp_cmd'] = pd.factorize(self.nids['ct_ftp_cmd'])[0]

        self.nids['srcip'] = preprocessing.LabelEncoder().fit_transform(self.nids['srcip'])
        self.nids['dstip'] = preprocessing.LabelEncoder().fit_transform(self.nids['dstip'])

#main function
def main():
    if len(sys.argv) < 4:
        print("Not enough arguments")
        return -1
    elif len(sys.argv) == 4:
        csv_name = sys.argv[1]
        c_method_name = sys.argv[2]
        task = sys.argv[3]
        
        print(csv_name)
        return 0
    elif len(sys.argv) == 5:
        csv_name = sys.argv[1]
        c_method_name = sys.argv[2]
        task = sys.argv[3]
        model_name = sys.argv[4]
        return 0
    
    print("Incorrect arguments")
    return -1
        
    
if __name__ == "__main__":
    main()