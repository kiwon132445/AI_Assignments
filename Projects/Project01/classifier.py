from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

class Classifier:
    
    ###########################################################################################################
    # Kiwon's section
    #
    # P1
    # Logistic Regression CV
    def logistic_init(self):
        self.log_reg = make_pipeline(StandardScaler(), LogisticRegressionCV(class_weight='balanced', scoring='balanced_accuracy'))
        
    def logistic_fit(self, x, y):
        self.log_reg.fit(x, y)
        
    def logistic_pred(self, x):
        return self.log_reg.predict(x)
    
    def logistic_score(self, x, y):
        return self.log_reg.score(x, y)
    
    def get_logistic(self):
        return self.log_reg
    
    #
    # P2
    #Support Vector Classification (SVC)
    def svc_init(self):
        self.svc = make_pipeline(StandardScaler(), SVC())
    
    def svc_fit(self, x, y):
        self.svc.fit(x, y)
        
    def svc_pred(self, x):
        return self.svc.predict(x)
    
    def svc_score(self, x, y):
        return self.svc.score(x, y)
    
    def get_svc(self):
        return self.svc
    
    ###########################################################################################################
    # June's section
    #
    # P1
    # Gaussian Naive Bayes (GaussianNB)
    def gauss_init(self):
        self.gauss = make_pipeline(RobustScaler(), GaussianNB())
    
    def gauss_fit(self, x, y):
        self.gauss.fit(x, y)
    
    def gauss_pred(self, x):
        self.gauss.pred(x)
    
    #
    # P2
    # Decision Tree Classifier
    def dtc_init(self):
        self.dtc = make_pipeline(StandardScaler(), DecisionTreeClassifier())
    
    def dtc_fit(self, x, y):
        self.dtc.fit(x, y)
    
    def dtc_pred(self, x):
        return self.dtc.pred(x)